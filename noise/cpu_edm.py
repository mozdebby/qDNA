import os
import pennylane as qml
from pennylane import numpy as np
from qiskit_aer import AerSimulator, Aer
from qiskit_aer.noise import NoiseModel
from fake_sherbrooke import FakeSherbrooke
import qiskit_aer
import random
import matplotlib.pyplot as plt
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import argparse
from collections import deque

os.environ['OPENBLAS_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser()

parser.add_argument('-length', type=int, help='length of the DNA sequence', required=True)
parser.add_argument('-num_data', type=int, help='number of data used in training and test', required=True)
parser.add_argument('-num_layer', type=int, help='number of layer in the circuit', required=True)
parser.add_argument('-num_epoch', type=int, help='the number of epochs', default=100)
parser.add_argument('-lr', type=int, help='learning rate of the optimizer', default=0.01)
parser.add_argument('-max_cpu', type=int, help='maximum number of cpu used', default=4)
parser.add_argument('-sys', type=int, help='system index', default=0)
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

backend = FakeSherbrooke()
noisy_sim = AerSimulator.from_backend(backend)
dev = qml.device("qiskit.aer", wires=n_qubit, backend= noisy_sim)
#dev = qml.device("qiskit.basicsim", wires=n_qubit)

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

def encoding_gates(input_string):
    for i, base in enumerate(input_string):
        if base == 'A':
            pass
        elif base == 'T':
            qml.RY(2 * np.arccos(-1 / np.sqrt(3)), wires=i)
        elif base == 'G':
            qml.RY(2 * np.arccos(-1 / np.sqrt(3)), wires=i)
            qml.PhaseShift(2 * np.pi / 3, wires=i)
        else:
            qml.RY(2 * np.arccos(-1 / np.sqrt(3)), wires=i)
            qml.PhaseShift(4 * np.pi / 3, wires=i)

def encoding_gates_dag(input_string):
    for i, base in enumerate(input_string):
        if base == 'A':
            pass
        elif base == 'T':
            qml.RY(-2 * np.arccos(-1 / np.sqrt(3)), wires=i)
        elif base == 'G':
            qml.PhaseShift(-2 * np.pi / 3, wires=i)
            qml.RY(-2 * np.arccos(-1 / np.sqrt(3)), wires=i)
        else:
            qml.PhaseShift(-4 * np.pi / 3, wires=i)
            qml.RY(-2 * np.arccos(-1 / np.sqrt(3)), wires=i)

def r_nxx_gate(params, n_qubit):
    for i in range(n_qubit-1):
        qml.CNOT(wires=[0, i+1])
    qml.RX(params, wires=0)
    for i in range(n_qubit-1):
        qml.CNOT(wires=[0, i+1]) 
        
def parameterized_circuit(params, n_qubit):
    for i in range(n_qubit):
        qml.RY(params[3 * 0], wires=i)
        qml.RZ(params[3 * 0+1], wires=i)
    r_nxx_gate(params[3 * 0+2], n_qubit)

def parameterized_circuit_dag(params, n_qubit):
    r_nxx_gate(-params[3 * 0], n_qubit)
    for i in range(n_qubit):
        qml.RZ(-params[3 * 0+1], wires=i)
        qml.RY(-params[3 * 0+2], wires=i)
            
class Model(torch.nn.Module):
    def __init__(self, num_qubit, num_layer, dev):
        super(Model, self).__init__()
        self.params = torch.nn.Parameter(torch.rand(3 * num_layer), requires_grad=True)
        self.num_qubit = num_qubit
        self.num_layer = num_layer
        self.qnode = qml.QNode(self.circuit, dev, interface='torch', diff_method='parameter-shift')
    
    def load_params(self, params):
        self.params = torch.nn.Parameter(torch.tensor(params))

    def read_params(self,):
        return np.array(self.params.detach().numpy())

    def circuit(self, params, input_string1, input_string2):
        n_qubit = self.num_qubit
        layer = self.num_layer
        for i in range(layer):
            parameterized_circuit(params[i*3:i*3+3], n_qubit)
            encoding_gates(input_string1)
        for i in range(layer):
            encoding_gates_dag(input_string2)
            parameterized_circuit_dag(torch.flip(params[(layer-i)*3-3:(layer-i)*3], dims=(-1,)), n_qubit)
        return qml.probs()


    def forward(self, input_string1, input_string2):
        return self.qnode(self.params, input_string1, input_string2)[0]



def loss_fn(model, input_string1, input_string2, length=length):
    pred = model.forward(input_string1, input_string2)
    similarity = calculate_edit_distance(input_string1, input_string2)
    similarity = 1 - similarity/length
    return (pred-similarity) ** 2

def loss_fn_set(model, data_list1, data_list2):
    loss_all = 0
    for i in range(len(data_list1)):
        data1 = data_list1[i]
        data2 = data_list2[i]
        loss_all = loss_all + loss_fn(model, data1, data2)
    loss_all = loss_all/len(data_list1)
    return loss_all

def abs_distance(model, data_list1, data_list2, length=length):
    loss_all = 0
    for i in range(len(data_list1)):
        data1 = data_list1[i]
        data2 = data_list2[i]
        pred = model.forward(data1, data2).detach().cpu().numpy()
        distance = calculate_edit_distance(data1, data2)
        loss_all = loss_all + np.abs(pred - 1 + distance/length)
    return loss_all/len(data_list1)

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
            loss = loss + 1
    return loss 

def train_worker(rank, world_size, lock, model, data_list_train1_seq, data_list_train2_seq, data_list_test1_seq, data_list_test2_seq, data_list_test3_seq, optimizer, steps, global_accuracy, global_max_accuracy, best_params):
    data_list_train1 = data_list_train1_seq[rank]
    data_list_train2 = data_list_train2_seq[rank]
    data_list_test1 = data_list_test1_seq[rank]
    data_list_test2 = data_list_test2_seq[rank]
    data_list_test3 = data_list_test3_seq[rank]
    acc = order_acc(model, data_list_test1, data_list_test2, data_list_test3)
    global_accuracy[0] += acc
    maximum_accuracy = global_accuracy[0]
    best_params = model.read_params()
    for epoch in range(steps):
        for i in range(num_data//world_size):    
            optimizer.zero_grad()
            loss = loss_fn(model, data_list_train1[i], data_list_train2[i])
            loss.backward()
            with lock:
                optimizer.step()
        acc = order_acc(model, data_list_test1, data_list_test2, data_list_test3)
        global_accuracy[epoch+1] += acc
        
        with lock:     
            if global_accuracy[epoch+1] > maximum_accuracy:
                maximum_accuracy = global_accuracy[epoch+1]
                best_params = model.read_params()
                print(f'best param now:{best_params}')
        print(f'cpu_{rank} processing: {100*epoch/steps}%')
    print(f'cpu_{rank} processing: 100%')
        
def train_generate(num_data):
    data_list_train1 = []
    data_list_train2 = []
    for i in range(num_data):
        data1 = generate_dna_sequence(length)
        data2 = generate_dna_sequence(length)
        data_list_train1.append(data1)
        data_list_train2.append(data2)
    return data_list_train1, data_list_train2

def test_generate(num_data):
    data_list_test1 = []
    data_list_test2 = []
    data_list_test3 = []
    for i in range(num_data):
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


if __name__ == "__main__":
    model = Model(n_qubit, num_layer, dev)
    model.share_memory()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0)
    
    data_list_train1, data_list_train2 = train_generate(num_data)
    data_list_test1, data_list_test2, data_list_test3 = test_generate(num_data)

    data_list_train1_sep = data_seperate(data_list_train1, world_size)
    data_list_train2_sep = data_seperate(data_list_train2, world_size)
    
    data_list_test1_sep = data_seperate(data_list_test1, world_size)
    data_list_test2_sep = data_seperate(data_list_test2, world_size)
    data_list_test3_sep = data_seperate(data_list_test3, world_size)
    
    with mp.Manager() as manager:
        lock = manager.Lock()
        global_accuracy = manager.list([0.0] * (steps+1))
        global_max_accuracy = manager.Value('d', float('inf'))
        best_params = manager.list(model.read_params())
        
        mp.spawn(
            train_worker,
            args=(world_size, lock, model, data_list_train1_sep, data_list_train2_sep, data_list_test1_sep, data_list_test2_sep, data_list_test3_sep, optimizer, steps, global_accuracy, global_max_accuracy, best_params),
            nprocs=world_size,
            join=True
        )
        
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