import os
import numpy as np
import random
import matplotlib.pyplot as plt
import multiprocessing as mp
import argparse
from collections import deque
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

os.environ['OPENBLAS_NUM_THREADS'] = '8'

parser = argparse.ArgumentParser()
parser.add_argument('-length',    type=int,   required=True,  help='length of the DNA sequence')
parser.add_argument('-num_data',  type=int,   required=True,  help='number of data points for training and test')
parser.add_argument('-num_layer', type=int,   required=True,  help='number of layers in the circuit')
parser.add_argument('-num_epoch', type=int,   default=100,    help='number of training epochs')
parser.add_argument('-lr',        type=float, default=0.01,   help='learning rate for SGD')
parser.add_argument('-max_cpu',   type=int,   default=4,      help='maximum number of CPUs to use')
parser.add_argument('-sys',       type=int,   default=0,      help='system/run index for output filenames')
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
alphabet   = "ACGT"

simulator = AerSimulator()

#Distance calculation

def all_moves(s):
    n, res = len(s), []
    for i in range(n):
        for j in range(i + 1, n + 1):
            block = s[i:j]
            rem   = s[:i] + s[j:]
            for k in range(len(rem) + 1):
                if k == i:
                    continue
                res.append(rem[:k] + block + rem[k:])
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
    #Bidirectional BFS; returns the graph distance between s and t
    if s == t:
        return 0
    if len(s) != len(t):
        raise ValueError("sequences have different lengths")
    
    front, back = {s: 0}, {t: 0}
    qf,    qb   = deque([s]), deque([t])

    while qf and qb:
        if len(qf) <= len(qb):
            cur = qf.popleft()
            d   = front[cur]
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
            d   = back[cur]
            for nb in neighbors(cur):
                if nb in back:
                    continue
                nd = d + 1
                if nb in front:
                    return nd + front[nb]
                back[nb] = nd
                qb.append(nb)
    return None

def calculate_edit_distance(seq1, seq2):
    return distance(seq1, seq2)

# Encoding / parameterised gate construction

_ENCODE_ANGLE = 2.0 * np.arccos(-1.0 / np.sqrt(3))

def encoding_gates(qc: QuantumCircuit, s: str) -> None:
    for j, base in enumerate(s):
        if base == 'A':
            pass                                        
        elif base == 'T':
            qc.ry(_ENCODE_ANGLE, j)
        elif base == 'G':
            qc.ry(_ENCODE_ANGLE, j)
            qc.p(2.0 * np.pi / 3.0, j)
        else:                                           
            qc.ry(_ENCODE_ANGLE, j)
            qc.p(4.0 * np.pi / 3.0, j)

def encoding_gates_dag(qc: QuantumCircuit, s: str) -> None:
    for j, base in enumerate(s):
        if base == 'A':
            pass
        elif base == 'T':
            qc.ry(-_ENCODE_ANGLE, j)
        elif base == 'G':
            qc.p(-2.0 * np.pi / 3.0, j)                
            qc.ry(-_ENCODE_ANGLE, j)
        else:                                           
            qc.p(-4.0 * np.pi / 3.0, j)
            qc.ry(-_ENCODE_ANGLE, j)

def r_nxx_gate(qc: QuantumCircuit, theta: float, n_qubit: int) -> None:
    for q in range(n_qubit - 1):
        qc.cx(0, q + 1)
    qc.rx(theta, 0)
    for q in range(n_qubit - 1):
        qc.cx(0, q + 1)

def parameterized_circuit(qc: QuantumCircuit, p: np.ndarray, n_qubit: int) -> None:
    for q in range(n_qubit):
        qc.ry(p[0], q)
        qc.rz(p[1], q)
    r_nxx_gate(qc, p[2], n_qubit)

def parameterized_circuit_dag(qc: QuantumCircuit, p: np.ndarray, n_qubit: int) -> None:
    r_nxx_gate(qc, -p[0], n_qubit)
    for q in range(n_qubit):
        qc.rz(-p[1], q)
        qc.ry(-p[2], q)

# Circuit execution

def build_circuit(
    params: np.ndarray,
    s1: str,
    s2: str,
    n_qubit: int,
    num_layer: int,
) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubit)

    # Forward pass
    for i in range(num_layer):
        p = params[i * 3 : i * 3 + 3]
        parameterized_circuit(qc, p, n_qubit)
        encoding_gates(qc, s1)

    # Backward pass
    for i in range(num_layer):
        encoding_gates_dag(qc, s2)
        raw_slice = params[(num_layer - i) * 3 - 3 : (num_layer - i) * 3]
        flipped   = raw_slice[::-1].copy()
        parameterized_circuit_dag(qc, flipped, n_qubit)
    return qc

def run_circuit(
    params: np.ndarray,
    s1: str,
    s2: str,
    n_qubit: int,
    num_layer: int,
) -> float:
    qc = build_circuit(params, s1, s2, n_qubit, num_layer)
    qc.measure_all()
    compiled_circuit = transpile(qc, simulator, optimization_level=0)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts()
    zero_state = "0" * n_qubit
    return counts.get(zero_state, 0) / 1000.0

# Per-sample worker functions

_FD_EPS = 1e-4   # finite-difference step size

def loss_fn(
    #Squared error between model output and normalised edit similarity
    params: np.ndarray,
    s1: str,
    s2: str,
    n_qubit: int,
    num_layer: int,
    seq_length: int,
) -> float:
    pred       = run_circuit(params, s1, s2, n_qubit, num_layer)
    similarity = 1.0 - calculate_edit_distance(s1, s2) / seq_length
    return (pred - similarity) ** 2

def compute_gradient(
    params: np.ndarray,
    s1: str,
    s2: str,
    n_qubit: int,
    num_layer: int,
    seq_length: int,
) -> np.ndarray:
    #Central finite-difference gradient of loss_fn
    grad = np.zeros(len(params), dtype=np.float64)
    for i in range(len(params)):
        p_plus  = params.copy(); p_plus[i]  += _FD_EPS
        p_minus = params.copy(); p_minus[i] -= _FD_EPS
        grad[i] = (
            loss_fn(p_plus,  s1, s2, n_qubit, num_layer, seq_length) -
            loss_fn(p_minus, s1, s2, n_qubit, num_layer, seq_length)
        ) / (2.0 * _FD_EPS)
    return grad

def abs_distance_metric(
    params: np.ndarray,
    data_list1: list,
    data_list2: list,
    n_qubit: int,
    num_layer: int,
    seq_length: int,
) -> float:
    total = 0.0
    for s1, s2 in zip(data_list1, data_list2):
        pred  = run_circuit(params, s1, s2, n_qubit, num_layer)
        d     = calculate_edit_distance(s1, s2)
        total += abs(pred - 1.0 + d / seq_length)
    return total / len(data_list1)

def order_acc(
    params: np.ndarray,
    data_list1: list,
    data_list2: list,
    data_list3: list,
    n_qubit: int,
    num_layer: int,
    seq_length: int,
) -> int:
    #Whether the model can rank sequences better
    correct = 0
    for s1, s2, s3 in zip(data_list1, data_list2, data_list3):
        d12   = calculate_edit_distance(s1, s2)
        d13   = calculate_edit_distance(s1, s3)
        sim12 = 1.0 - d12 / seq_length
        sim13 = 1.0 - d13 / seq_length
        p12   = run_circuit(params, s1, s2, n_qubit, num_layer)
        p13   = run_circuit(params, s1, s3, n_qubit, num_layer)
        if ((sim12 - sim13) < 0) == ((p12 - p13) < 0):
            correct += 1
    return correct


def _shared_to_numpy(shared_arr: mp.Array) -> np.ndarray:
    """Numpy view directly into a mp.Array shared-memory buffer (no copy)."""
    return np.frombuffer(shared_arr.get_obj(), dtype=np.float64)

def train_worker(
    rank: int,
    world_size: int,
    lock,
    shared_params: mp.Array,
    train1: list,
    train2: list,
    test1: list,
    test2: list,
    test3: list,
    steps: int,
    global_accuracy,           
    global_max_accuracy,       
    best_params_store: list,   
    n_qubit: int,
    num_layer: int,
    seq_length: int,
    lr: float,
) -> None:

    #initial accuracy
    params = _shared_to_numpy(shared_params).copy()
    acc    = order_acc(params, test1, test2, test3, n_qubit, num_layer, seq_length)
    with lock:
        global_accuracy[0] = global_accuracy[0] + acc

    #training/evaluation
    for epoch in range(steps):

        # training
        for s1, s2 in zip(train1, train2):
            with lock:
                params = _shared_to_numpy(shared_params).copy()
            grad = compute_gradient(params, s1, s2, n_qubit, num_layer, seq_length)
            with lock:
                view  = _shared_to_numpy(shared_params)
                view -= lr * grad       

        # evaluation
        params = _shared_to_numpy(shared_params).copy()
        acc    = order_acc(params, test1, test2, test3, n_qubit, num_layer, seq_length)
        with lock:
            global_accuracy[epoch + 1] = global_accuracy[epoch + 1] + acc
            if global_accuracy[epoch + 1] > global_max_accuracy.value:
                global_max_accuracy.value = global_accuracy[epoch + 1]
                for idx, v in enumerate(params):
                    best_params_store[idx] = float(v)
                print(f'[rank {rank}] new best accuracy={global_max_accuracy.value:.4f} '
                      f'params={params}')

        print(f'cpu_{rank}: {100 * (epoch + 1) / steps:.1f}% complete')

    print(f'cpu_{rank}: 100% complete')

# Data generation

def generate_dna_sequence(length):
    return ''.join(random.choice('ATGC') for _ in range(length))

def train_generate(num_data: int, length: int):
    s1 = [generate_dna_sequence(length) for _ in range(num_data)]
    s2 = [generate_dna_sequence(length) for _ in range(num_data)]
    return s1, s2

def test_generate(num_data: int, length: int):
    s1 = [generate_dna_sequence(length) for _ in range(num_data)]
    s2 = [generate_dna_sequence(length) for _ in range(num_data)]
    s3 = [generate_dna_sequence(length) for _ in range(num_data)]
    return s1, s2, s3

def data_split(data: list, world_size: int) -> list:
    chunk = len(data) // world_size
    return [data[i * chunk : (i + 1) * chunk] for i in range(world_size)]

# Main

if __name__ == "__main__":
    n_params    = 3 * num_layer
    init_params = np.random.rand(n_params)

    #Shared memory for circuit parameters
    shared_params = mp.Array('d', init_params)

    data_train1, data_train2             = train_generate(num_data, length)
    data_test1,  data_test2, data_test3  = test_generate(num_data, length)

    train1_sep = data_split(data_train1, world_size)
    train2_sep = data_split(data_train2, world_size)
    test1_sep  = data_split(data_test1,  world_size)
    test2_sep  = data_split(data_test2,  world_size)
    test3_sep  = data_split(data_test3,  world_size)

    with mp.Manager() as manager:
        lock                = manager.Lock()
        global_accuracy     = manager.list([0.0] * (steps + 1))
        global_max_accuracy = manager.Value('d', float('-inf'))
        best_params_store   = manager.list([float(v) for v in init_params])

        processes = []
        for rank in range(world_size):
            p = mp.Process(
                target=train_worker,
                args=(
                    rank, world_size, lock, shared_params,
                    train1_sep[rank], train2_sep[rank],
                    test1_sep[rank],  test2_sep[rank], test3_sep[rank],
                    steps, global_accuracy, global_max_accuracy, best_params_store,
                    n_qubit, num_layer, length, lr,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        global_accuracy = np.array(list(global_accuracy)) / num_data
        best_params     = np.array(list(best_params_store))

    # save results
    folder_name = (
        f"mp_result_length_{length}_num_data_{num_data}"
        f"_num_layer_{num_layer}_steps_{steps}_lr_{lr}"
    )
    os.makedirs(folder_name, exist_ok=True)

    np.save(os.path.join(folder_name, f'acc_{sys_idx}.npy'),    global_accuracy)
    np.save(os.path.join(folder_name, f'params_{sys_idx}.npy'), best_params)

    plt.figure(figsize=(10, 6))
    plt.title('Order Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Order Accuracy')
    plt.plot(global_accuracy)
    plt.savefig(os.path.join(folder_name, f'oc_{sys_idx}.png'))
    plt.close()
