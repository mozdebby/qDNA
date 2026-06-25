import os
import numpy as np
import random
import matplotlib.pyplot as plt
import multiprocessing as mp
import argparse
import pandas as pd
import time
from collections import deque
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime.fake_provider import FakeCasablancaV2, FakeSherbrooke

os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['RAYON_NUM_THREADS'] = '1'   # ← stops Rayon spawning threads per process
os.environ['OMP_NUM_THREADS'] = '1'     # ← same for OpenMP (also used by Aer internals


parser = argparse.ArgumentParser()
parser.add_argument('-length',    type=int,   required=True,  help='length of the DNA sequence')
parser.add_argument('-num_data',  type=int,   required=True,  help='number of data points for training and test')
parser.add_argument('-num_layer', type=int,   required=True,  help='number of layers in the circuit')
parser.add_argument('-num_epoch', type=int,   default=100,    help='number of training epochs')
parser.add_argument('-lr',        type=float, default=0.01,   help='learning rate for SGD')
parser.add_argument('-max_cpu',   type=int,   default=4,      help='maximum number of CPUs to use')
parser.add_argument('-sys',       type=int,   default=0,      help='system/run index for output filenames')
parser.add_argument('-shots',       type=int,   default=1,      help='how many shots to run (for value 1 it takes the ideal quantum state instead)')
parser.add_argument('-noisemodel',      type=int,       default=0,      help='whether to use the sherbrooke noise model (0 means disabled)')
parser.add_argument('-gradient_method', type=str, default='fd', choices=['fd', 'parameter_shift'], 
                    help="gradient method: 'fd' (central finite-difference, default) or 'parameter_shift' (exact analytic gradient, same number of circuit evals as fd but no step-size bias)")

args = parser.parse_args()

length        = args.length
num_data      = args.num_data
num_layer     = args.num_layer
n_qubit       = length
steps         = args.num_epoch
lr            = args.lr
maximum_world = args.max_cpu
sys_idx       = args.sys
shots         = args.shots 
with_noise    = args.noisemodel
gradient_method = args.gradient_method

world_size = min(maximum_world, mp.cpu_count())
alphabet   = "ACGT"

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

class Circuit:    
    def __init__(self, num_layer: int, n_qubit: int, shots: int, with_noise):
        self._ENCODE_ANGLE = 2.0 * np.arccos(-1.0 / np.sqrt(3))
        self.qc = QuantumCircuit(n_qubit)
        self.theta = ParameterVector('theta', 3*num_layer)
        self.base = ParameterVector('base', n_qubit*2*2) #two gates for two sequences
        self.shots = shots
        self.noisy = (with_noise != 0)
        if with_noise == 1: self.noise_model = 'fake_backend'
        else: self.noise_model =  'aer'
        self.__build_circuit(n_qubit, num_layer)
        
    def __encoding_gates(self, n_qubit: int) -> None:
        for i in range(n_qubit):
            self.qc.ry(self.base[i*2], i) 
            self.qc.p(self.base[i*2+1], i)        

    def __encoding_gates_dag(self, n_qubit: int) -> None:
        start_idx = n_qubit*2
        for i in range(n_qubit):
            self.qc.p(-self.base[start_idx+i*2+1], i) 
            self.qc.ry(-self.base[start_idx+i*2], i) 

    def __r_nxx_gate(self, theta, n_qubit: int) -> None:
        for q in range(n_qubit - 1):
            self.qc.cx(0, q + 1)
        self.qc.rx(theta, 0)
        for q in range(n_qubit - 1):
            self.qc.cx(0, q + 1)

    def __parameterized_circuit(self, idx: int, n_qubit: int) -> None:
        for q in range(n_qubit):
            self.qc.ry(self.theta[idx], q)
            self.qc.rz(self.theta[idx+1], q)
        self.__r_nxx_gate(self.theta[idx+2], n_qubit)

    def __parameterized_circuit_dag(self, idx: int, n_qubit: int) -> None:
        self.__r_nxx_gate(-self.theta[idx+2], n_qubit)
        for q in range(n_qubit):
            self.qc.rz(-self.theta[idx+1], q)
            self.qc.ry(-self.theta[idx], q)


    def __encode_string_angles(self, s1: str):
        baseangles = []
        for i, base in enumerate(s1):
            idx = i*2
            if base == 'A':
                baseangles.append(0)
                baseangles.append(0)
            elif base == 'T':
                baseangles.append(self._ENCODE_ANGLE)
                baseangles.append(0)
            elif base == 'G':
                baseangles.append(self._ENCODE_ANGLE)
                baseangles.append(2.0 * np.pi / 3.0)
            else:
                baseangles.append(self._ENCODE_ANGLE)
                baseangles.append(4.0 * np.pi / 3.0)        
        return baseangles
    
    def __input_strings(self, s1: str, s2: str):
        return self.__encode_string_angles(s1) + self.__encode_string_angles(s2)

    def __build_circuit(
        self,
        n_qubit: int,
        num_layer: int,        
    ):
        # Forward pass        
        for i in range(num_layer):
            idx = i*3
            self.__parameterized_circuit(idx, n_qubit)
            self.__encoding_gates(n_qubit)

        # Backward pass
        for i in range(num_layer):
            idx = num_layer*3 - i*3 - 3
            self.__encoding_gates_dag(n_qubit)
            self.__parameterized_circuit_dag(idx, n_qubit)
        
        # if self.shots > 1 and with_noise == 0:
        #     self.qc.measure_all()
        #     self.simulator = AerSimulator()
        #     self.qc = transpile(self.qc, self.simulator)
        
        if self.noisy:
            if self.noise_model == 'fake_backend':
                #start = time.perf_counter()
                self.qc.measure_all()
                self.backend = FakeSherbrooke()
                self.qc = transpile(self.qc, self.backend, optimization_level=3)
            else:
                self.qc.measure_all()
                p1=0.001 
                p2=0.01
                noise_model = NoiseModel()
                error_1q = depolarizing_error(p1, 1)
                error_2q = depolarizing_error(p2, 2)
                noise_model.add_all_qubit_quantum_error(error_1q, ['ry', 'rz', 'rx', 'p'])
                noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
                self.backend = AerSimulator(noise_model=noise_model)
                self.qc = transpile(self.qc, self.backend, optimization_level=3)
                return noise_model

            
            
            


    def run_circuit(
        self,
        params: np.ndarray,
        s1: str,
        s2: str,
    ) -> float:
        
        #bound_qc is a copy of the circuit with the parameter values in place 
        bound_qc = self.qc.assign_parameters(dict(zip(self.theta, params)))
        bound_qc = bound_qc.assign_parameters(dict(zip(self.base, self.__input_strings(s1, s2))))
        start = time.perf_counter()
        if self.shots == 1:
            sv = Statevector.from_instruction(bound_qc)
            print(f"Run_circuit complete ({time.perf_counter()-start} s)")
            return float(sv.probabilities()[0])
        elif not self.noisy:
            sv = Statevector.from_instruction(bound_qc)
            counts = sv.sample_counts(self.shots)
            print(f"Run_circuit complete ({time.perf_counter()-start} s)")
            return counts.get("0" * bound_qc.num_qubits, 0) / self.shots
            # job = self.simulator.run(self.qc, shots=self.shots)
            # result = job.result()
            # counts = result.get_counts()
            # zero_state = "0" * n_qubit
            # return counts.get(zero_state, 0) / self.shots
        else:            
            #print(f"Running {self.shots} shots...")
            start = time.perf_counter()
            result = self.backend.run(bound_qc, shots=self.shots).result()
            counts = result.get_counts()
            zero_state = "0" * n_qubit
            print(f"Run_circuit complete ({time.perf_counter()-start} s)")
            return counts.get(zero_state, 0) / self.shots
            

# Per-sample worker functions

_FD_EPS = 1e-4   # finite-difference step size

def loss_fn(
    #Squared error between model output and normalised edit similarity
    qc: Circuit,
    params: np.ndarray,
    s1: str,
    s2: str,
    edm: int,
    n_qubit: int,
    num_layer: int,
    seq_length: int,
) -> float:
    pred       = qc.run_circuit(params, s1, s2)
    similarity = 1.0 - edm / seq_length
    return (pred - similarity) ** 2

def compute_gradient(
    qc: Circuit,
    params: np.ndarray,
    s1: str,
    s2: str,
    edm: int,
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
            loss_fn(qc, p_plus,  s1, s2, edm, n_qubit, num_layer, seq_length) -
            loss_fn(qc, p_minus, s1, s2, edm, n_qubit, num_layer, seq_length)
        ) / (2.0 * _FD_EPS)
    return grad

_SHIFT = np.pi / 2.0  # standard parameter-shift offset for Pauli-rotation gates (ry/rz/rx)

def compute_gradient_parameter_shift(
    qc: Circuit,
    params: np.ndarray,
    s1: str,
    s2: str,
    edm: int,
    n_qubit: int,
    num_layer: int,
    seq_length: int,
    ) -> np.ndarray:
    #Exact gradient of loss_fn via the parameter-shift rule.
    #Every theta[i] feeds only ry/rz/rx gates (eigenvalues +-1), including the
    #mirrored "-theta[i]" use in the dagger pass, so the standard +-pi/2 shift
    #rule applies directly to qc.run_circuit's output (the prediction itself).
    similarity = 1.0 - edm / seq_length
    pred = qc.run_circuit(params, s1, s2)  # reused for the chain-rule factor below

    grad = np.zeros(len(params), dtype=np.float64)
    for i in range(len(params)):
        p_plus  = params.copy(); p_plus[i]  += _SHIFT
        p_minus = params.copy(); p_minus[i] -= _SHIFT
        pred_plus  = qc.run_circuit(p_plus,  s1, s2)
        pred_minus = qc.run_circuit(p_minus, s1, s2)
        dpred_dtheta_i = (pred_plus - pred_minus) / 2.0

        # chain rule: loss = (pred - similarity)^2  =>  dloss/dtheta_i = 2*(pred-similarity)*dpred/dtheta_i
        grad[i] = 2.0 * (pred - similarity) * dpred_dtheta_i
    return grad

# def abs_distance_metric(
#     params: np.ndarray,
#     data_list1: list,
#     data_list2: list,
#     n_qubit: int,
#     num_layer: int,
#     seq_length: int,
# ) -> float:
#     total = 0.0
#     for s1, s2 in zip(data_list1, data_list2):
#         pred  = run_circuit(params, s1, s2, n_qubit, num_layer)
#         d     = calculate_edit_distance(s1, s2)
#         total += abs(pred - 1.0 + d / seq_length)
#     return total / len(data_list1)

def order_acc(
    qc: Circuit,
    params: np.ndarray,
    data_list1: list,
    data_list2: list,
    data_list3: list,
    edm_12: list,
    edm_13: list,
    n_qubit: int,
    num_layer: int,
    seq_length: int,
) -> int:
    #Whether the model can rank sequences better
    correct = 0
    for s1, s2, s3, edm12, edm13  in zip(data_list1, data_list2, data_list3, edm_12, edm_13):
        d12   = edm12
        d13   = edm13
        sim12 = 1.0 - d12 / seq_length
        sim13 = 1.0 - d13 / seq_length
        p12   = qc.run_circuit(params, s1, s2)
        p13   = qc.run_circuit(params, s1, s3)
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
    edm_train: list,
    test1: list,
    test2: list,
    test3: list,
    edm_test12: list,
    edm_test13: list,
    steps: int,
    global_accuracy,           
    global_max_accuracy,       
    best_params_store: list,   
    n_qubit: int,
    num_layer: int,
    seq_length: int,
    lr: float,
    shots: int,
    with_noise,
    gradient_method: str = 'fd',
) -> None:
    qc = Circuit(num_layer=num_layer, n_qubit=n_qubit, shots=shots, with_noise=with_noise)
    grad_fn = compute_gradient_parameter_shift if gradient_method == 'parameter_shift' else compute_gradient
    print(f"Building circuit complete ({rank}). Param length: {len(shared_params)}")

    #initial accuracy
    params = _shared_to_numpy(shared_params).copy()
    acc    = order_acc(qc, params, test1, test2, test3, edm_test12, edm_test13, n_qubit, num_layer, seq_length)
    with lock:
        global_accuracy[0] = global_accuracy[0] + acc

    #print(f"Starting training ({rank}).")
    #training/evaluations
    for epoch in range(steps):

        # training
        for s1, s2, edm in zip(train1, train2, edm_train):
            #print(f"Training step worker ({rank})")
            with lock:
                params = _shared_to_numpy(shared_params).copy()
            grad = grad_fn(qc, params, s1, s2, edm, n_qubit, num_layer, seq_length) #compute_gradient(qc, params, s1, s2, edm, n_qubit, num_layer, seq_length)
            with lock:
                view  = _shared_to_numpy(shared_params)
                view -= lr * grad       

        # evaluation
        params = _shared_to_numpy(shared_params).copy()
        acc    = order_acc(qc, params, test1, test2, test3, edm_test12, edm_test13, n_qubit, num_layer, seq_length)
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

    df = pd.read_feather("trainingdata7_1200.feather")
    data_train1 = df.iloc[:num_data ,0].to_numpy()
    data_train2 = df.iloc[:num_data ,1].to_numpy()
    data_edm = df.iloc[:num_data ,2].to_numpy()

    #Read training data
    df = pd.read_feather("testingdata7_1200.feather")
    data_test1 = df.iloc[:num_data ,0].to_numpy()
    data_test2 = df.iloc[:num_data ,1].to_numpy()
    data_test3 = df.iloc[:num_data ,2].to_numpy()
    data_edm12 = df.iloc[:num_data ,3].to_numpy()
    data_edm13 = df.iloc[:num_data ,4].to_numpy()

    train1_sep = data_split(data_train1, world_size)
    train2_sep = data_split(data_train2, world_size)
    edm_train_sep = data_split(data_edm, world_size)
    test1_sep  = data_split(data_test1,  world_size)
    test2_sep  = data_split(data_test2,  world_size)
    test3_sep  = data_split(data_test3,  world_size)
    edm_test12_sep = data_split(data_edm12,  world_size)
    edm_test13_sep = data_split(data_edm13,  world_size)

    print("Data generation finished.")
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
                    train1_sep[rank], train2_sep[rank], edm_train_sep[rank],
                    test1_sep[rank],  test2_sep[rank], test3_sep[rank], 
                    edm_test12_sep[rank], edm_test13_sep[rank],
                    steps, global_accuracy, global_max_accuracy, best_params_store,
                    n_qubit, num_layer, length, lr, shots, with_noise, gradient_method,
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
        f"_num_layer_{num_layer}_steps_{steps}_lr_{lr}_shots_{shots}_noise_{with_noise}_gradient_{gradient_method}"
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
