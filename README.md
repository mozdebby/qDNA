## Organisation

The files at the highest level are the relevant files, currently qiskit_dna_similarity_noiseless.py, qiskit_dna_similarity_measnoise.py and the folder "results". The folder "olderfiles' is a dump for previous approaches and their results.

## Running the simulation

To run the file qiskit_dna_error.py, have both .feather files in the same directory and use the following command:

noiseless: `qiskit_dna_error.py -length 7 -num_data 1200 -num_layer 12`
with measurement error: '-shots 10000'
with a noise model of a fake backend: '-noisemodel 1' 
with a simulated noise model: '-noisemodel 2' 

This file has a bug for multiprocessing that annihilates learning. This bug has been identified and is being fixed. You can adjust the amount of processing cores used with `-max_cpu [number of cores]` (default=4) but this will worsen learning.
