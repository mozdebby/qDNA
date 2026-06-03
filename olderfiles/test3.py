import os
import random
import numpy as np
import torch
import torch.optim as optim
import qiskit
from torch.nn import Parameter
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Parameter as QiskitParameter
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit.primitives import StatevectorSampler
from collections import deque
import matplotlib.pyplot as plt
import argparse
import multiprocessing as mp
from multiprocessing import Manager, Lock

os.environ['OPENBLAS_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('-length', type=int, required=True, help='length of the DNA sequence')
parser.add_argument('-num_data', type=int, required=True, help='number of data used in training and test')
parser.add_argument('-num_layer', type=int, required=True, help='number of layer in the circuit')
parser.add_argument('-num_epoch', type=int, default=100, help='the number of epochs')
parser.add_argument('-lr', type=float, default=0.01, help='learning rate of the optimizer')
parser.add_argument('-max_cpu', type=int, default=4, help='maximum number of cpu used')
parser.add_argument('-sys', type=int, default=0, help='system index')
args = parser.parse_args()

length = args.length
num_data = args.num_data
num_layer = args.num_layer
n_qubit = length
steps = args.num_epoch
lr = args.lr
maximum_world = args.max_cpu
sys = args.sys

world_size = min(maximum_world, mp.cpu_count())

alphabet = "ACGT"

# --- DNA sequence utilities ---
def all_moves(s):
    n = len(s)
    res = []
    for i in range(n):
        for j in range(i+1, n+1):
            block = s[i:j]
            rem = s[:i] + s[j:]
            for k in range(len(rem)+1):
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
                res.append(s[:idx] + a + s[idx+1:])
    return res

def neighbors(s):
    return all_moves(s) + all_substitutions(s)

def distance(s, t):
    if s == t:
        return 0
    if len(s) != len(t):
        raise ValueError("different length")
    front = {s: 0}
    back = {t: 0}
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

# --- Qiskit encoding gates ---
def encoding_gates(qc, input_string):
    for i, base in enumerate(input_string):
        if base == 'A':
            pass
        elif base == 'T':
            qc.ry(2 * np.arccos(-1 / np.sqrt(3)), i)
        elif base == 'G':
            qc.ry(2 * np.arccos(-1 / np.sqrt(3)), i)
            qc.p(2 * np.pi / 3, i)
        else:  # 'C'
            qc.ry(2 * np.arccos(-1 / np.sqrt(3)), i)
            qc.p(4 * np.pi / 3, i)

def encoding_gates_dag(qc, input_string):
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

# --- Qiskit parameterized circuit ---
def r_nxx_gate(qc, param, n_qubit):
    for i in range(n_qubit-1):
        qc.cx(0, i+1)
    qc.rx(param, 0)
    for i in range(n_qubit-1):
        qc.cx(0, i+1)

def parameterized_circuit(qc, params, n_qubit, layer_idx):
    params_np = params.detach().numpy() if torch.is_tensor(params) else params
    for i in range(n_qubit):
        qc.ry(float(params_np[3 * layer_idx]), i)
        qc.rz(float(params_np[3 * layer_idx + 1]), i)
    r_nxx_gate(qc, float(params_np[3 * layer_idx + 2]), n_qubit)

def parameterized_circuit_dag(qc, params, n_qubit, layer_idx):
    params_np = params.detach().numpy() if torch.is_tensor(params) else params
    r_nxx_gate(qc, float(-params_np[3 * (num_layer - layer_idx - 1)]), n_qubit)
    for i in range(n_qubit):
        qc.rz(float(-params_np[3 * (num_layer - layer_idx - 1) + 1]), i)
        qc.ry(float(-params_np[3 * (num_layer - layer_idx - 1) + 2]), i)

# --- Qiskit Model ---
class QiskitModel(torch.nn.Module):
    def __init__(self, num_qubit, num_layer):
        super(QiskitModel, self).__init__()
        self.params = Parameter(torch.rand(3 * num_layer), requires_grad=True)
        self.num_qubit = num_qubit
        self.num_layer = num_layer
        self.sampler = StatevectorSampler()

    def circuit(self, params, input_string1, input_string2):
        qr = QuantumRegister(self.num_qubit)
        cr = ClassicalRegister(1, name='cr') #todo: waarom 1? is niet niet n bits?
        qc = QuantumCircuit(qr, cr)

        # Convert PyTorch tensor to NumPy array
        params_np = params.detach().numpy() if torch.is_tensor(params) else params

        # Encoding and parameterized layers
        for i in range(self.num_layer):
            parameterized_circuit(qc, params_np, self.num_qubit, i)
            encoding_gates(qc, input_string1)
        for i in range(self.num_layer):
            encoding_gates_dag(qc, input_string2)
            parameterized_circuit_dag(qc, params_np, self.num_qubit, i)

        qc.measure(qr[0], cr[0])
        return qc

    def forward(self, input_string1, input_string2):
        params_np = self.params.detach().numpy()
        qc = self.circuit(params_np, input_string1, input_string2)
        job = self.sampler.run([qc], shots=1024)
        result = job.result()
        counts = result[0].data.cr.get_counts()
        #counts = result.quasi_dists[0]
        probs = np.array([counts.get(bin(i), 0) for i in range(2)])
        return torch.tensor(probs[0], dtype=torch.float32, requires_grad=True)

# --- Loss and metrics ---
def loss_fn(model, input_string1, input_string2, length=length):
    pred = model.forward(input_string1, input_string2)
    similarity = calculate_edit_distance(input_string1, input_string2)
    similarity = 1 - similarity/length
    return (pred - similarity) ** 2

def loss_fn_set(model, data_list1, data_list2):
    loss_all = 0
    for i in range(len(data_list1)):
        data1 = data_list1[i]
        data2 = data_list2[i]
        loss_all += loss_fn(model, data1, data2)
    return loss_all / len(data_list1)

def abs_distance(model, data_list1, data_list2, length=length):
    loss_all = 0
    for i in range(len(data_list1)):
        data1 = data_list1[i]
        data2 = data_list2[i]
        pred = model.forward(data1, data2).detach().cpu().numpy()
        distance = calculate_edit_distance(data1, data2)
        loss_all += np.abs(pred - 1 + distance/length)
    return loss_all / len(data_list1)

def order_acc(model, data_list1, data_list2, data_list3, length=length):
    loss = 0
    for i in range(len(data_list1)):
        data1 = data_list1[i]
        data2 = data_list2[i]
        data3 = data_list3[i]
        distance12 = calculate_edit_distance(data1, data2)
        distance13 = calculate_edit_distance(data1, data3)
        similarity12 = 1 - distance12 / length
        similarity13 = 1 - distance13 / length
        pred12 = model.forward(data1, data2).detach().cpu().numpy()
        pred13 = model.forward(data1, data3).detach().cpu().numpy()
        similarity = (similarity12 - similarity13) < 0
        pred = (pred12 - pred13) < 0
        if similarity == pred:
            loss += 1
    return loss

# --- Data generation ---
def train_generate(num_data):
    data_list_train1 = []
    data_list_train2 = []
    for _ in range(num_data):
        data1 = generate_dna_sequence(length)
        data2 = generate_dna_sequence(length)
        data_list_train1.append(data1)
        data_list_train2.append(data2)
    return data_list_train1, data_list_train2

def test_generate(num_data):
    data_list_test1 = []
    data_list_test2 = []
    data_list_test3 = []
    for _ in range(num_data):
        data1 = generate_dna_sequence(length)
        data2 = generate_dna_sequence(length)
        data3 = generate_dna_sequence(length)
        data_list_test1.append(data1)
        data_list_test2.append(data2)
        data_list_test3.append(data3)
    return data_list_test1, data_list_test2, data_list_test3

def data_seperate(data_list, world_size):
    chunk_size = len(data_list) // world_size
    data_list_sep = []
    for i in range(world_size):
        data_list_sep.append(data_list[chunk_size * i: chunk_size * i + chunk_size])
    return data_list_sep

# --- Training worker ---
def train_worker(rank, world_size, lock, model, data_list_train1, data_list_train2, data_list_test1, data_list_test2, data_list_test3, optimizer, steps, global_accuracy, global_max_accuracy, best_params):
    acc = order_acc(model, data_list_test1, data_list_test2, data_list_test3)
    with lock:
        global_accuracy[0] += acc
    maximum_accuracy = global_accuracy[0]
    best_params[:] = model.params.detach().numpy().tolist()

    for epoch in range(steps):
        for i in range(len(data_list_train1)):
            optimizer.zero_grad()
            loss = loss_fn(model, data_list_train1[i], data_list_train2[i])
            loss.backward()
            with lock:
                optimizer.step()
        acc = order_acc(model, data_list_test1, data_list_test2, data_list_test3)
        with lock:
            global_accuracy[epoch+1] += acc
            if global_accuracy[epoch+1] > maximum_accuracy:
                maximum_accuracy = global_accuracy[epoch+1]
                best_params[:] = model.params.detach().numpy().tolist()
                print(f'best param now: {best_params}')
        print(f'cpu_{rank} processing: {100*epoch/steps}%')
    print(f'cpu_{rank} processing: 100%')

if __name__ == "__main__":
    model = QiskitModel(n_qubit, num_layer)
    model.share_memory()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0)

    data_list_train1, data_list_train2 = train_generate(num_data)
    data_list_test1, data_list_test2, data_list_test3 = test_generate(num_data)

    data_list_train1_sep = data_seperate(data_list_train1, world_size)
    data_list_train2_sep = data_seperate(data_list_train2, world_size)
    data_list_test1_sep = data_seperate(data_list_test1, world_size)
    data_list_test2_sep = data_seperate(data_list_test2, world_size)
    data_list_test3_sep = data_seperate(data_list_test3, world_size)

    with Manager() as manager:
        lock = Lock()
        global_accuracy = manager.list([0.0] * (steps+1))
        global_max_accuracy = manager.Value('d', float('inf'))
        best_params = manager.list(model.params.detach().numpy().tolist())

        processes = []
        for rank in range(world_size):
            p = mp.Process(
                target=train_worker,
                args=(rank, world_size, lock, model, data_list_train1_sep[rank], data_list_train2_sep[rank],
                      data_list_test1_sep[rank], data_list_test2_sep[rank], data_list_test3_sep[rank],
                      optimizer, steps, global_accuracy, global_max_accuracy, best_params)
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        global_accuracy = np.array(global_accuracy)
        global_accuracy = global_accuracy / num_data

        folder_name = f"mp_result_length_{length}_num_data_{num_data}_num_layer_{num_layer}_steps_{steps}_lr_{lr}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        np.save(os.path.join(folder_name, f'acc_{sys}.npy'), global_accuracy)
        np.save(os.path.join(folder_name, f'params_{sys}.npy'), best_params)

        plt.figure(figsize=(10, 6))
        plt.title('Order Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Order Accuracy')
        plt.plot(global_accuracy)
        plt.savefig(os.path.join(folder_name, f'oc_{sys}.png'))