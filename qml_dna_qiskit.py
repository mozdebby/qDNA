#!/usr/bin/env python3
import os
import random
import argparse
import multiprocessing as mp
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

os.environ['OPENBLAS_NUM_THREADS'] = '1'

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-length',    type=int,   required=True,
                    help='DNA sequence length (also sets number of qubits)')
parser.add_argument('-num_data',  type=int,   required=True,
                    help='number of sequences used for training and testing')
parser.add_argument('-num_layer', type=int,   required=True,
                    help='number of variational layers in the circuit')
parser.add_argument('-num_epoch', type=int,   default=100,
                    help='max COBYLA iterations (replaces gradient epochs)')
parser.add_argument('-max_cpu',   type=int,   default=4,
                    help='max worker processes for Pool')
parser.add_argument('-sys',       type=int,   default=0,
                    help='run index used in output file names')
# -lr is accepted for CLI compatibility but COBYLA is gradient-free
parser.add_argument('-lr',        type=float, default=0.01,
                    help='(ignored — COBYLA does not use a learning rate)')
args = parser.parse_args()

length     = args.length
num_data   = args.num_data
num_layer  = args.num_layer
n_qubit    = length          
steps      = args.num_epoch
maximum_world = min(args.max_cpu, mp.cpu_count())
sys    = args.sys

alphabet = "ACGT"

#Distance calculation

def all_moves(s: str):
    n = len(s)
    res = []
    for i in range(n):
        for j in range(i + 1, n + 1):
            block = s[i:j]
            rem   = s[:i] + s[j:]
            for k in range(len(rem) + 1):
                if k != i:
                    res.append(rem[:k] + block + rem[k:])
    return res


def all_substitutions(s: str):
    res = []
    for idx, ch in enumerate(s):
        for a in alphabet:
            if a != ch:
                res.append(s[:idx] + a + s[idx+1:])
    return res


def neighbors(s: str):
    return all_moves(s) + all_substitutions(s)


def distance(s: str, t: str):
    #Bidirectional BFS; returns the graph distance between s and t
    if s == t:
        return 0
    if len(s) != len(t):
        raise ValueError("different length")
    
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

def calculate_edit_distance(s1: str, s2: str) -> int:
    return distance(s1, s2)


# Encoding / parameterised gate construction

_ENC_ANGLE = 2.0 * np.arccos(-1.0 / np.sqrt(3.0))


def encoding_gates(qc: QuantumCircuit, seq: str) -> None:
    for i, base in enumerate(seq):
        if base == 'T':
            qc.ry(_ENC_ANGLE, i)
        elif base == 'G':
            qc.ry(_ENC_ANGLE, i)
            qc.p(2.0 * np.pi / 3.0, i)
        elif base == 'C':
            qc.ry(_ENC_ANGLE, i)
            qc.p(4.0 * np.pi / 3.0, i)


def encoding_gates_dag(qc: QuantumCircuit, seq: str) -> None:
    for i, base in enumerate(seq):
        if base == 'T':
            qc.ry(-_ENC_ANGLE, i)
        elif base == 'G':
            qc.p(-2.0 * np.pi / 3.0, i)
            qc.ry(-_ENC_ANGLE, i)
        elif base == 'C':
            qc.p(-4.0 * np.pi / 3.0, i)
            qc.ry(-_ENC_ANGLE, i)


def r_nxx_gate(qc: QuantumCircuit, param: np.ndarray, nq: int) -> None:
    for q in range(nq - 1):
        qc.cx(0, q + 1)
    qc.rx(float(-param[0]), 0)
    for q in range(nq - 1):
        qc.cx(0, q + 1)

def parameterized_circuit(qc: QuantumCircuit, param: np.ndarray, nq: int) -> None:    
    for q in range(nq):
        qc.ry(float(param[0]), q)
        qc.rz(float(param[1]), q)
    # r_nxx: entangle all qubits through qubit 0
    r_nxx_gate(qc, param, nq)

def parameterized_circuit_dag(qc: QuantumCircuit, param_flip: np.ndarray, nq: int) -> None:
    r_nxx_gate
    for q in range(nq):
        qc.rz(float(-param_flip[1]), q)
        qc.ry(float(-param_flip[2]), q)

# Circuit execution

def run_circuit(params: np.ndarray, s1: str, s2: str,
                nq: int, nl: int) -> float:
    #Output close to 1 when s1 ≈ s2 and close to 0 when they are distant.
    
    qc = QuantumCircuit(nq)

    #Forward pass
    for li in range(nl):
        parameterized_circuit(qc, params[li * 3: li * 3 + 3], nq)
        encoding_gates(qc, s1)

    #Backward pass
    for li in range(nl):
        encoding_gates_dag(qc, s2)
        p_flip = params[(nl - li - 1) * 3: (nl - li) * 3][::-1]
        parameterized_circuit_dag(qc, p_flip, nq)

    # Qiskit statevector simulation
    probs = Statevector(qc).probabilities()
    return float(probs[0])   # P(|00…0⟩)

# Per-sample worker functions

def loss_fn(args):
    #Squared error between model output and normalised edit similarity
    params, s1, s2, nq, nl, ln = args
    pred = run_circuit(params, s1, s2, nq, nl)
    sim  = 1.0 - calculate_edit_distance(s1, s2) / ln
    return (pred - sim) ** 2

# def loss_fn_set(model, data_list1, data_list2):
#     loss_all = 0
#     for i in range(len(data_list1)):
#         data1 = data_list1[i]
#         data2 = data_list2[i]
#         loss_all = loss_all + loss_fn(model, data1, data2)
#     loss_all = loss_all/len(data_list1)
#     return loss_all

def order_acc(args):
    #Whether the model can rank sequences better
    #--Each process does this once, originally this function looped
    #--Maybe the inequality in the return should be <= instead of <
    params, s1, s2, s3, nq, nl, ln = args
    sim12 = 1.0 - calculate_edit_distance(s1, s2) / ln
    sim13 = 1.0 - calculate_edit_distance(s1, s3) / ln
    p12   = run_circuit(params, s1, s2, nq, nl)
    p13   = run_circuit(params, s1, s3, nq, nl)
    return 1 if (sim12 <= sim13) == (p12 <= p13) else 0


# Data generation

def generate_dna_sequence(length):
    return ''.join(random.choice('ATGC') for _ in range(length))

def train_generate(n: int, ln: int):
    data_list_train1 = []
    data_list_train2 = []
    for i in range(num_data):
        data1 = generate_dna_sequence(length)
        data2 = generate_dna_sequence(length)
        data_list_train1.append(data1)
        data_list_train2.append(data2)
    return data_list_train1, data_list_train2


def test_generate(n: int, ln: int):
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

# def data_seperate(data_list, world_size):
#     chunk_size = len(data_list) // world_size
#     data_list_sep = []
#     for i in range(world_size):
#         data_list_sep.append(data_list[chunk_size * i: chunk_size * i + chunk_size])
#     return data_list_sep

# Main

if __name__ == '__main__':
    print(f"Config: length={length}, num_data={num_data}, num_layer={num_layer}, "
          f"steps={steps}, workers={maximum_world}")

    train1, train2        = train_generate(num_data, length)
    test1,  test2, test3  = test_generate(num_data, length)

    #Initial parameters, randomised
    params0 = np.random.rand(3 * num_layer)

    #Shared state data
    state: dict = dict(
        accs        = [],          # order-accuracy per COBYLA iteration
        best_params = params0.copy(),
        best_acc    = -1,
        iteration   = 0,
    )

    with mp.Pool(processes=maximum_world) as pool:

        def objective(params: np.ndarray) -> float:
            #Defining the job
            job_args = [
                (params, train1[i], train2[i], n_qubit, num_layer, length)
                for i in range(num_data)
            ]
            return float(np.mean(pool.map(loss_fn, job_args)))

        def callback(params: np.ndarray) -> None:
            #Evaluate accuracy
            job_args = [
                (params, test1[i], test2[i], test3[i], n_qubit, num_layer, length)
                for i in range(num_data)
            ]
            acc = sum(pool.map(order_acc, job_args))
            state['iteration'] += 1
            state['accs'].append(acc / num_data)

            if acc > state['best_acc']:
                state['best_acc']    = acc
                state['best_params'] = params.copy()
                print(f"-iter {state['iteration']:4d}  "
                      f"NEW BEST  acc={acc}/{num_data}")

            print(f"  iter {state['iteration']:4d}/{steps}  "
                  f"acc={acc}/{num_data}")

        minimize(
            objective,
            params0,
            method='COBYLA',
            callback=callback,
            options=dict(
                maxiter = steps,   # max COBYLA iterations
                rhobeg  = 0.5,     # initial trust-region radius
                catol   = 0.0,     # no inequality constraints
            ),
        )

    # Save results
    global_accuracy        = np.array(state['accs'])
    best_params = state['best_params']

    folder = (f"result_length{length}_data{num_data}"
              f"_layers{num_layer}_steps{steps}")
    os.makedirs(folder, exist_ok=True)

    np.save(os.path.join(folder, f'acc_{sys}.npy'),    global_accuracy)
    np.save(os.path.join(folder, f'params_{sys}.npy'), best_params)

    plt.figure(figsize=(10, 6))
    plt.title('Order Accuracy')
    plt.xlabel('COBYLA Iteration')
    plt.ylabel('Order Accuracy')
    plt.plot(global_accuracy)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f'oc_{sys}.png'))
    plt.close()

    print(f"\nDone. Results saved to '{folder}/'")
    print(f"Best ordering accuracy: {state['best_acc']}/{num_data}")
