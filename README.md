## Organisation

The files at the highest level are the relevant files, currently qiskit_dna_similarity_noiseless.py, qiskit_dna_similarity_measnoise.py and the folder "results". The folder "olderfiles' is a dump for previous approaches and their results.

## Running the simulation

A recommended setting for running the noiseless script: 

`qiskit_dna_similarity_noiseless.py -length 8  -num_data 1200 -num_layer 12 -num_epoch 100`

This file has a bug for multiprocessing that annihilates learning. This bug has been identified and is being fixed. You can adjust the amount of processing cores used with `-max_cpu [number of cores]` (default=4) but this will worsen learning.
