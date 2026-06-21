import os
import numpy as np
import random
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import shared_memory
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
parser.add_argument('-meas_noise', type=bool, default=False, help='measurement noise (true) or perfect simulation (false)')
parser.add_argument('-sys',       type=int,   default=0,      help='system/run index for output filenames')
args = parser.parse_args()

length        = args.length
num_data      = args.num_data
num_layer     = args.num_layer
n_qubit       = length
steps         = args.num_epoch
lr            = args.lr
maximum_world = args.max_cpu
meas_noise    = args.meas_noise
sys_idx       = args.sys

world_size = min(maximum_world, mp.cpu_count())
alphabet   = "ACGT"

if (meas_noise):
    simulator = AerSimulator()

# ---------------------------------------------------------------------------
# Edit-distance helpers (unchanged from original)
# ---------------------------------------------------------------------------

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

# def all_block_transposition(s):
#     res = []
#     n for 

def neighbors(s):
    return all_moves(s) + all_substitutions(s)

def distance(s, t):
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

def generate_dna_sequence(length):
    return ''.join(random.choice('ATGC') for _ in range(length))

def calculate_edit_distance(seq1, seq2):
    return distance(seq1, seq2)

# ---------------------------------------------------------------------------
# Quantum circuit (Qiskit)
# ---------------------------------------------------------------------------

# Pre-compute the encoding angle used for T/G/C bases.
_ENCODE_ANGLE = 2.0 * np.arccos(-1.0 / np.sqrt(3))

def _apply_encoding(qc: QuantumCircuit, s: str) -> None:
    """Encode a DNA string onto the qubits (forward direction)."""
    for j, base in enumerate(s):
        if base == 'A':
            pass                                        # |0> -> no gate needed
        elif base == 'T':
            qc.ry(_ENCODE_ANGLE, j)
        elif base == 'G':
            qc.ry(_ENCODE_ANGLE, j)
            qc.p(2.0 * np.pi / 3.0, j)
        else:                                           # 'C'
            qc.ry(_ENCODE_ANGLE, j)
            qc.p(4.0 * np.pi / 3.0, j)

def _apply_encoding_dag(qc: QuantumCircuit, s: str) -> None:
    """Adjoint (inverse) of the DNA encoding layer."""
    for j, base in enumerate(s):
        if base == 'A':
            pass
        elif base == 'T':
            qc.ry(-_ENCODE_ANGLE, j)
        elif base == 'G':
            qc.p(-2.0 * np.pi / 3.0, j)                # phase before RY (reversed order)
            qc.ry(-_ENCODE_ANGLE, j)
        else:                                           # 'C'
            qc.p(-4.0 * np.pi / 3.0, j)
            qc.ry(-_ENCODE_ANGLE, j)

def _apply_r_nxx(qc: QuantumCircuit, theta: float, n_qubit: int) -> None:
    """Multi-qubit RXX-style gate: fan-out CNOTs, RX on qubit 0, fan-in CNOTs."""
    for q in range(n_qubit - 1):
        qc.cx(0, q + 1)
    qc.rx(theta, 0)
    for q in range(n_qubit - 1):
        qc.cx(0, q + 1)

def _apply_param_layer(qc: QuantumCircuit, p: np.ndarray, n_qubit: int) -> None:
    """Single parameterized layer: RY + RZ on all qubits, then R_NXX gate."""
    for q in range(n_qubit):
        qc.ry(p[0], q)
        qc.rz(p[1], q)
    _apply_r_nxx(qc, p[2], n_qubit)

def _apply_param_layer_dag(qc: QuantumCircuit, p: np.ndarray, n_qubit: int) -> None:
    """Adjoint of a parameterized layer.

    Matches the original parameterized_circuit_dag which receives already-flipped
    params and maps index 0 -> R_NXX angle, 1 -> RZ angle, 2 -> RY angle (all negated).
    """
    _apply_r_nxx(qc, -p[0], n_qubit)
    for q in range(n_qubit):
        qc.rz(-p[1], q)
        qc.ry(-p[2], q)

def build_circuit(
    params: np.ndarray,
    s1: str,
    s2: str,
    n_qubit: int,
    num_layer: int,
) -> QuantumCircuit:
    """Build the full quantum circuit for two input DNA strings."""
    qc = QuantumCircuit(n_qubit)
    # Forward layers: parameterized block then encode s1
    for i in range(num_layer):
        p = params[i * 3 : i * 3 + 3]
        _apply_param_layer(qc, p, n_qubit)
        _apply_encoding(qc, s1)

    # Backward layers: decode s2 (dag) then parameterized block (dag, reversed order)
    for i in range(num_layer):
        _apply_encoding_dag(qc, s2)
        # Mirror torch.flip on the corresponding 3-element slice
        raw_slice = params[(num_layer - i) * 3 - 3 : (num_layer - i) * 3]
        flipped   = raw_slice[::-1].copy()
        _apply_param_layer_dag(qc, flipped, n_qubit)

    return qc

def run_circuit(
    params: np.ndarray,
    s1: str,
    s2: str,
    n_qubit: int,
    num_layer: int,
) -> float:
    """Execute the circuit and return the probability of the all-zeros state."""
    qc = build_circuit(params, s1, s2, n_qubit, num_layer)
    if meas_noise:
        qc.measure_all()
        compiled_circuit = transpile(qc, simulator, optimization_level=0)
        job = simulator.run(compiled_circuit, shots=1000)
        result = job.result()
        counts = result.get_counts()
        zero_state = "0" * n_qubit
        return counts.get(zero_state, 0) / 1000.0
    else:
        sv = Statevector.from_instruction(qc)
        return float(sv.probabilities()[0])

# ---------------------------------------------------------------------------
# Loss and gradients  (central finite differences — no autograd needed)
# ---------------------------------------------------------------------------

_FD_EPS = 1e-4   # finite-difference step size

def loss_fn(
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
    """Central finite-difference gradient of loss_fn w.r.t. all parameters."""
    grad = np.zeros(len(params), dtype=np.float64)
    for i in range(len(params)):
        p_plus  = params.copy(); p_plus[i]  += _FD_EPS
        p_minus = params.copy(); p_minus[i] -= _FD_EPS
        grad[i] = (
            loss_fn(p_plus,  s1, s2, n_qubit, num_layer, seq_length) -
            loss_fn(p_minus, s1, s2, n_qubit, num_layer, seq_length)
        ) / (2.0 * _FD_EPS)
    return grad

# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

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
    """Count how often the model's pairwise similarity ordering matches ground truth."""
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

# ---------------------------------------------------------------------------
# Shared-memory helpers
# ---------------------------------------------------------------------------

# def _shared_to_numpy(shared_arr: mp.Array) -> np.ndarray:
#     """Numpy view directly into a mp.Array shared-memory buffer (no copy)."""
#     return np.frombuffer(shared_arr.get_obj(), dtype=np.float64)

# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def train_worker(
    rank: int,
    world_size: int,
    num_data: int,
    barrier,
    shared_params,
    results_acc,
    results_param,
    train1: list,
    train2: list,
    test1: list,
    test2: list,
    test3: list,
    steps: int,
    global_accuracy,           # Manager.list
    global_max_accuracy,       # Manager.Value
    best_params_store: list,   # Manager.list
    n_qubit: int,
    num_layer: int,
    seq_length: int,
    lr: float,
) -> None:

    # ---- initial accuracy ----
    #params = _shared_to_numpy(shared_params).copy()
    acc    = order_acc(shared_params, test1, test2, test3, n_qubit, num_layer, seq_length)
    results_acc[rank] = acc

    datasize = num_data // world_size
    idx = datasize * rank
    #barrier.wait()
    # ---- per-epoch SGD ----
    for epoch in range(steps):
        # Per-sample SGD step
        for s1, s2 in zip(train1[idx:idx+datasize], train2[idx:idx+datasize]):
            local_params = np.array(shared_params, copy=True)
            grad = compute_gradient(local_params, s1, s2, n_qubit, num_layer, seq_length)
            local_params -= lr * grad
            results_param[rank] = local_params
            barrier.wait() # Wait for all other workers
            barrier.wait() # wait for parent to update data
            # with lock:
            #     view  = _shared_to_numpy(shared_params)
            #     view -= lr * grad          # in-place SGD on shared memory

        # # Evaluate and track best params
        # params = _shared_to_numpy(shared_params).copy()
        # acc    = order_acc(params, test1, test2, test3, n_qubit, num_layer, seq_length)
        # with lock:
        #     global_accuracy[epoch + 1] = global_accuracy[epoch + 1] + acc
        #     if global_accuracy[epoch + 1] > global_max_accuracy.value:
        #         global_max_accuracy.value = global_accuracy[epoch + 1]
        #         for idx, v in enumerate(params):
        #             best_params_store[idx] = float(v)
        #         print(f'[rank {rank}] new best accuracy={global_max_accuracy.value:.4f} '
        #               f'params={params}')

        # Return results to parent
        results_acc[rank] = order_acc(local_params, test1[idx:idx+datasize], test2[idx:idx+datasize], test3[idx:idx+datasize], n_qubit, num_layer, seq_length)
        #results_param[rank] = local_params
        #print(f'cpu_{rank}: {100 * (epoch + 1) / steps:.1f}% complete')
        barrier.wait() # Wait for all other workers
        # barrier.wait() # wait for parent to update data


    print(f'cpu_{rank}: 100% complete')

# ---------------------------------------------------------------------------
# Data generation / splitting
# ---------------------------------------------------------------------------

def generate_random_dna2(num_data: int, length: int):
    s1 = [generate_dna_sequence(length) for _ in range(num_data)]
    s2 = [generate_dna_sequence(length) for _ in range(num_data)]
    return np.array(s1, dtype="U8"), np.array(s2, dtype="U8")

def generate_random_dna3(num_data: int, length: int):
    s1 = [generate_dna_sequence(length) for _ in range(num_data)]
    s2 = [generate_dna_sequence(length) for _ in range(num_data)]
    s3 = [generate_dna_sequence(length) for _ in range(num_data)]
    return np.array(s1, dtype="U8"), np.array(s2, dtype="U8"), np.array(s3, dtype="U8")

def data_split(data: list, world_size: int) -> list:
    chunk = len(data) // world_size
    return [data[i * chunk : (i + 1) * chunk] for i in range(world_size)]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    n_params    = 3 * num_layer
    init_params = np.random.rand(n_params)
    # Shared memory for circuit parameters — writable by all worker processes.
    #shared_params = mp.Array('d', init_params)

    data_train1_gen, data_train2_gen                = generate_random_dna2(num_data, length)
    data_test1_gen,  data_test2_gen, data_test3_gen = generate_random_dna3(num_data, length)



    # train1_sep = data_split(data_train1, world_size)
    # train2_sep = data_split(data_train2, world_size)
    # test1_sep  = data_split(data_test1,  world_size)
    # test2_sep  = data_split(data_test2,  world_size)
    # test3_sep  = data_split(data_test3,  world_size)
    #todo: shared data, pass indices for range


    with mp.Manager() as manager:
        barrier                = manager.Barrier(world_size+1)
        global_accuracy     = manager.list([0.0] * (steps + 1))
        global_max_accuracy = manager.Value('d', float('-inf'))
        best_params_store   = manager.list([float(v) for v in init_params])

        shm_params = shared_memory.SharedMemory(create=True, size=init_params.nbytes)
        shared_params = np.ndarray(init_params.shape, dtype=init_params.dtype, buffer=shm_params.buf)
        np.copyto(shared_params, init_params)

        shm_train1 = shared_memory.SharedMemory(create=True, size=data_train1_gen.nbytes)
        shm_train2 = shared_memory.SharedMemory(create=True, size=data_train2_gen.nbytes)
        shm_test1 = shared_memory.SharedMemory(create=True, size=data_test1_gen.nbytes)
        shm_test2 = shared_memory.SharedMemory(create=True, size=data_test2_gen.nbytes)
        shm_test3 = shared_memory.SharedMemory(create=True, size=data_test3_gen.nbytes)
        
        data_train1 = np.ndarray(data_train1_gen.shape, dtype=data_train1_gen.dtype, buffer=shm_train1.buf)
        data_train2 = np.ndarray(data_train2_gen.shape, dtype=data_train2_gen.dtype, buffer=shm_train2.buf)
        data_test1 = np.ndarray(data_test1_gen.shape, dtype=data_test1_gen.dtype, buffer=shm_test1.buf)
        data_test2 = np.ndarray(data_test2_gen.shape, dtype=data_test2_gen.dtype, buffer=shm_test2.buf)
        data_test3 = np.ndarray(data_test3_gen.shape, dtype=data_test3_gen.dtype, buffer=shm_test3.buf)

        np.copyto(data_train1, data_train1_gen)
        np.copyto(data_train2, data_train2_gen)
        np.copyto(data_test1_gen, data_test1)
        np.copyto(data_test2_gen, data_test2)
        np.copyto(data_test3_gen, data_test3)

        shm_list = [shm_params, shm_train1, shm_train2, shm_test1, shm_test2, shm_test3]

        results_acc = manager.dict()
        results_param = manager.dict()

        processes = []
        for rank in range(world_size):
            p = mp.Process(
                target=train_worker,
                args=(rank, world_size, num_data, barrier, shared_params, results_acc, results_param,
                      data_train1, data_train2, data_test1, data_test2, data_test3, 
                      steps, global_accuracy, global_max_accuracy, best_params_store, n_qubit, num_layer, length, lr),
            )
            p.start()
            processes.append(p)

        datasize = num_data // world_size
        for i in range(steps):
            for j in range(datasize): #adjust after results for each calculation
                #Wait for all workers to finish one training              
                
                barrier.wait()

                # Collect all results
                all_params = [results_param[rank] for rank in range(world_size)]

                ground_params = np.ndarray(init_params.shape, dtype=init_params.dtype)
                np.copyto(ground_params, shared_params)
                for params in results_param:
                    shared_params -= ground_params - params
                shared_params /= world_size
                print(ground_params)
                print(shared_params)
                print("-------------")
                # # Compute average and update shared_params
                # avg_params = np.mean(all_params, axis=0)
                # for k in range(len(shared_params)):
                #     shared_params[k] = avg_params[k]

                barrier.wait()
            #get order accuracy scores
            max_order_acc = np.max(results_acc)
            avg_order_acc = np.mean(results_acc)

            print(f'{100 * (i + 1) / steps:.1f}% complete. Order accuracy mean-max: {max_order_acc}-{max_order_acc}')

        for p in processes:
            p.join()


        global_accuracy = np.array(list(global_accuracy)) / num_data
        best_params     = np.array(list(best_params_store))
        #shared_params.shm.close()
        #shared_params.shm.unlink()
        for i in shm_list:
            i.close()
            i.unlink()

    # ---- Save results ----
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
