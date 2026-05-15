import os
import random
import argparse
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector

os.environ['OPENBLAS_NUM_THREADS'] = '1'

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-length',   type=int, required=True,  help='length of the DNA sequence')
parser.add_argument('-num_data', type=int, required=True,  help='number of data used in training and test')
parser.add_argument('-num_layer',type=int, required=True,  help='number of layers in the circuit')
parser.add_argument('-num_epoch',type=int, default=100,    help='number of epochs')
parser.add_argument('-lr',       type=float, default=0.01, help='learning rate')   # changed type int→float
parser.add_argument('-max_cpu',  type=int, default=4,      help='maximum number of CPUs used')
parser.add_argument('-sys',      type=int, default=0,      help='system index')
args = parser.parse_args()

length        = args.length
num_data      = args.num_data
num_layer     = args.num_layer
n_qubit       = length
steps         = args.num_epoch
lr            = args.lr
maximum_world = args.max_cpu
sys_idx       = args.sys

world_size = min(maximum_world, mp.cpu_count())

alphabet = "ACGT"

# ---------------------------------------------------------------------------
# Edit-distance helpers  (unchanged from original)
# ---------------------------------------------------------------------------

def all_moves(s):
    n = len(s)
    res = []
    for i in range(n):
        for j in range(i + 1, n + 1):
            block = s[i:j]
            rem = s[:i] + s[j:]
            for k in range(len(rem) + 1):
                if k == i:
                    continue
                t = rem[:k] + block + rem[k:]
                res.append(t)
    return res

def all_substitutions(s):
    res = []
    for idx, ch in enumerate(s):
        for a in alphabet:
            if a != ch:
                res.append(s[:idx] + a + s[idx + 1:])
    return res

def neighbors(s):
    return all_moves(s) + all_substitutions(s)

def distance(s, t):
    if s == t:
        return 0
    if len(s) != len(t):
        raise ValueError("different length")
    front = {s: 0}
    back  = {t: 0}
    qf = deque([s])
    qb = deque([t])
    while qf and qb:
        if len(qf) <= len(qb):
            cur = qf.popleft()
            d = front[cur]
            for nb in neighbors(cur):
                if nb in front:
                    continue
                nd = d + 1
                if nb in back:
                    return nd + back[nb]
                front[nb] = nd
                qf.append(nb)
        else:
            cur = qb.popleft()
            d = back[cur]
            for nb in neighbors(cur):
                if nb in back:
                    continue
                nd = d + 1
                if nb in front:
                    return nd + front[nb]
                back[nb] = nd
                qb.append(nb)
    return None

def generate_dna_sequence(length):
    return ''.join(random.choice('ATGC') for _ in range(length))

def calculate_edit_distance(seq1, seq2):
    return distance(seq1, seq2)

# ---------------------------------------------------------------------------
# Qiskit circuit builders
# ---------------------------------------------------------------------------

def add_encoding_gates(qc: QuantumCircuit, input_string: str):
    """Encode DNA bases onto qubits (tetrahedral encoding)."""
    for i, base in enumerate(input_string):
        if base == 'A':
            pass  # |0⟩, no gate needed
        elif base == 'T':
            qc.ry(2 * np.arccos(-1 / np.sqrt(3)), i)
        elif base == 'G':
            qc.ry(2 * np.arccos(-1 / np.sqrt(3)), i)
            qc.p(2 * np.pi / 3, i)          # PhaseShift
        else:  # 'C'
            qc.ry(2 * np.arccos(-1 / np.sqrt(3)), i)
            qc.p(4 * np.pi / 3, i)          # PhaseShift

def add_encoding_gates_dag(qc: QuantumCircuit, input_string: str):
    """Adjoint of encoding (reversed gate order, negated angles)."""
    for i, base in enumerate(input_string):
        if base == 'A':
            pass
        elif base == 'T':
            qc.ry(-2 * np.arccos(-1 / np.sqrt(3)), i)
        elif base == 'G':
            qc.p(-2 * np.pi / 3, i)
            qc.ry(-2 * np.arccos(-1 / np.sqrt(3)), i)
        else:  # 'C'
            qc.p(-4 * np.pi / 3, i)
            qc.ry(-2 * np.arccos(-1 / np.sqrt(3)), i)

def add_r_nxx_gate(qc: QuantumCircuit, theta, n_qubit: int):
    """
    R_NXX(θ): CNOT ladder from qubit 0 to all others, RX on qubit 0, CNOT ladder back.
    `theta` may be a float or a Qiskit Parameter.
    """
    for i in range(1, n_qubit):
        qc.cx(0, i)
    qc.rx(theta, 0)
    for i in range(1, n_qubit):
        qc.cx(0, i)

def add_parameterized_circuit(qc: QuantumCircuit, params, n_qubit: int):
    """One forward layer: RY+RZ on all qubits, then R_NXX."""
    for i in range(n_qubit):
        qc.ry(params[0], i)
        qc.rz(params[1], i)
    add_r_nxx_gate(qc, params[2], n_qubit)

def add_parameterized_circuit_dag(qc: QuantumCircuit, params, n_qubit: int):
    """Adjoint of one layer (reversed, negated)."""
    add_r_nxx_gate(qc, -params[0], n_qubit)
    for i in range(n_qubit):
        qc.rz(-params[1], i)
        qc.ry(-params[2], i)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Model(torch.nn.Module):
    """
    Quantum model backed by Qiskit statevector simulation.

    The circuit is rebuilt on every forward pass with the current parameter
    values so that PyTorch autograd can differentiate through it via the
    parameter-shift rule implemented manually.  For simplicity we use the
    finite-difference / direct statevector approach: bind concrete float
    values and return prob[0] as a plain tensor.

    NOTE: gradients flow through `self.params` via PyTorch's autograd graph
    because we use `torch.func` / custom Function below.
    """

    def __init__(self, num_qubit: int, num_layer: int):
        super().__init__()
        self.params    = torch.nn.Parameter(torch.rand(3 * num_layer), requires_grad=True)
        self.num_qubit = num_qubit
        self.num_layer = num_layer

    def load_params(self, params):
        self.params = torch.nn.Parameter(torch.tensor(params, dtype=torch.float32))

    def read_params(self):
        return self.params.detach().numpy().copy()

    def _build_circuit(self, param_values: np.ndarray,
                       input_string1: str, input_string2: str) -> QuantumCircuit:
        """Construct the full circuit with concrete (bound) parameter values."""
        n   = self.num_qubit
        lay = self.num_layer
        qc  = QuantumCircuit(n)

        # Forward half
        for i in range(lay):
            p = param_values[i * 3: i * 3 + 3]
            add_parameterized_circuit(qc, p, n)
            add_encoding_gates(qc, input_string1)

        # Reverse half (adjoint)
        for i in range(lay):
            add_encoding_gates_dag(qc, input_string2)
            # mirrors original: params[(layer-i)*3-3 : (layer-i)*3] flipped
            raw = param_values[(lay - i) * 3 - 3: (lay - i) * 3]
            p   = raw[::-1]                          # flip within the 3-element slice
            add_parameterized_circuit_dag(qc, p, n)

        return qc

    def _prob_zero(self, param_values: np.ndarray,
                   input_string1: str, input_string2: str) -> float:
        """Return P(|00...0⟩) for the given circuit as a plain Python float."""
        qc = self._build_circuit(param_values, input_string1, input_string2)
        sv = Statevector(qc)
        # Probability of the all-zeros computational basis state
        return float(abs(sv.data[0]) ** 2)

    def forward(self, input_string1: str, input_string2: str) -> torch.Tensor:
        """
        Forward pass with parameter-shift gradient support.

        We wrap the statevector call in a custom autograd Function so that
        PyTorch can back-propagate through the quantum circuit using the
        parameter-shift rule  (shift = π/2).
        """
        return _QuantumFunction.apply(self.params, self, input_string1, input_string2)


class _QuantumFunction(torch.autograd.Function):
    """Custom autograd Function implementing the parameter-shift rule."""

    @staticmethod
    def forward(ctx, params: torch.Tensor, model: Model,
                s1: str, s2: str) -> torch.Tensor:
        pv = params.detach().numpy()
        val = model._prob_zero(pv, s1, s2)
        ctx.save_for_backward(params)
        ctx.model = model
        ctx.s1 = s1
        ctx.s2 = s2
        return torch.tensor(val, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        params, = ctx.saved_tensors
        model   = ctx.model
        s1, s2  = ctx.s1, ctx.s2
        pv      = params.detach().numpy()
        shift   = np.pi / 2
        grads   = np.zeros_like(pv)
        for k in range(len(pv)):
            pv_p = pv.copy(); pv_p[k] += shift
            pv_m = pv.copy(); pv_m[k] -= shift
            grads[k] = 0.5 * (model._prob_zero(pv_p, s1, s2)
                               - model._prob_zero(pv_m, s1, s2))
        grad_tensor = torch.tensor(grads, dtype=torch.float32) * grad_output
        # gradients for: params, model (None), s1 (None), s2 (None)
        return grad_tensor, None, None, None

# ---------------------------------------------------------------------------
# Loss / metric functions  (logic unchanged, adapted for new model API)
# ---------------------------------------------------------------------------

def loss_fn(model: Model, s1: str, s2: str) -> torch.Tensor:
    pred       = model.forward(s1, s2)
    similarity = 1.0 - calculate_edit_distance(s1, s2) / length
    return (pred - similarity) ** 2

def loss_fn_set(model: Model, list1, list2) -> torch.Tensor:
    total = sum(loss_fn(model, a, b) for a, b in zip(list1, list2))
    return total / len(list1)

def abs_distance(model: Model, list1, list2) -> float:
    total = 0.0
    for a, b in zip(list1, list2):
        pred = model.forward(a, b).detach().numpy()
        d    = calculate_edit_distance(a, b)
        total += abs(pred - 1 + d / length)
    return total / len(list1)

def order_acc(model: Model, list1, list2, list3) -> int:
    correct = 0
    for a, b, c in zip(list1, list2, list3):
        d12 = calculate_edit_distance(a, b)
        d13 = calculate_edit_distance(a, c)
        s12 = 1 - d12 / length
        s13 = 1 - d13 / length
        p12 = model.forward(a, b).detach().numpy()
        p13 = model.forward(a, c).detach().numpy()
        if ((s12 - s13) < 0) == ((p12 - p13) < 0):
            correct += 1
    return correct

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def train_generate(n):
    l1, l2 = [], []
    for _ in range(n):
        l1.append(generate_dna_sequence(length))
        l2.append(generate_dna_sequence(length))
    return l1, l2

def test_generate(n):
    l1, l2, l3 = [], [], []
    for _ in range(n):
        l1.append(generate_dna_sequence(length))
        l2.append(generate_dna_sequence(length))
        l3.append(generate_dna_sequence(length))
    return l1, l2, l3

def data_separate(data, ws):
    chunk = len(data) // ws
    return [data[chunk * i: chunk * i + chunk] for i in range(ws)]

# ---------------------------------------------------------------------------
# Training worker
# ---------------------------------------------------------------------------

def train_worker(rank, world_size, lock, model,
                 train1_sep, train2_sep,
                 test1_sep,  test2_sep,  test3_sep,
                 optimizer, steps,
                 global_accuracy, global_max_accuracy, best_params):

    tr1, tr2 = train1_sep[rank], train2_sep[rank]
    te1, te2, te3 = test1_sep[rank], test2_sep[rank], test3_sep[rank]

    acc = order_acc(model, te1, te2, te3)
    global_accuracy[0] += acc
    maximum_accuracy = global_accuracy[0]

    for epoch in range(steps):
        for a, b in zip(tr1, tr2):
            optimizer.zero_grad()
            loss = loss_fn(model, a, b)
            loss.backward()
            with lock:
                optimizer.step()

        acc = order_acc(model, te1, te2, te3)
        global_accuracy[epoch + 1] += acc

        with lock:
            if global_accuracy[epoch + 1] > maximum_accuracy:
                maximum_accuracy = global_accuracy[epoch + 1]
                new_best = model.read_params()
                for i, v in enumerate(new_best):
                    best_params[i] = v
                print(f'best params now: {new_best}')

        print(f'cpu_{rank} processing: {100 * epoch / steps:.1f}%')

    print(f'cpu_{rank} processing: 100%')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = Model(n_qubit, num_layer)
    model.share_memory()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0)

    train1, train2           = train_generate(num_data)
    test1,  test2,  test3    = test_generate(num_data)

    train1_sep = data_separate(train1, world_size)
    train2_sep = data_separate(train2, world_size)
    test1_sep  = data_separate(test1,  world_size)
    test2_sep  = data_separate(test2,  world_size)
    test3_sep  = data_separate(test3,  world_size)

    with mp.Manager() as manager:
        lock               = manager.Lock()
        global_accuracy    = manager.list([0.0] * (steps + 1))
        global_max_accuracy= manager.Value('d', float('inf'))
        best_params        = manager.list(model.read_params().tolist())

        mp.spawn(
            train_worker,
            args=(world_size, lock, model,
                  train1_sep, train2_sep,
                  test1_sep, test2_sep, test3_sep,
                  optimizer, steps,
                  global_accuracy, global_max_accuracy, best_params),
            nprocs=world_size,
            join=True,
        )

        global_accuracy = np.array(list(global_accuracy)) / num_data

        folder = (f"mp_result_length_{length}_num_data_{num_data}"
                  f"_num_layer_{num_layer}_steps_{steps}_lr_{lr}")
        os.makedirs(folder, exist_ok=True)

        np.save(os.path.join(folder, f'acc_{sys_idx}.npy'),    global_accuracy)
        np.save(os.path.join(folder, f'params_{sys_idx}.npy'), list(best_params))

        plt.figure(figsize=(10, 6))
        plt.title('Order Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Order Accuracy')
        plt.plot(global_accuracy)
        plt.savefig(os.path.join(folder, f'oc_{sys_idx}.png'))
        plt.close()
