from sys import stderr, exit, argv
from os.path import isfile, isdir, realpath, dirname, exists
import numpy as np
from scipy.io import loadmat
# Loading spiketimes for entorhinal cortex recording

neuron_index = int(argv[1])
rec_length = argv[2]

if len(argv) > 3:
    data_path = argv[3]
else:
    CODE_DIR = '{}/..'.format(dirname(realpath(__file__)))
    data_path = '{}/data'.format(CODE_DIR)

DATA_DIR = '{}/CA1'.format(data_path)

validNeurons = np.load(
    '{}/validNeurons.npy'.format(DATA_DIR)).astype(int)
neuron = validNeurons[neuron_index]

spiketimes = np.load(
        '{}/spks/spiketimes_neuron{}.npy'.format(DATA_DIR, neuron))

if rec_length == '40min':
    T_rec = 2400.
if rec_length == '90min':
    T_rec = 5400.

# Add 5 seconds to make sure that only spikes with sufficient spiking history are considered
T_0 = spiketimes[0] + 5.

spiketimes = spiketimes - T_0
spiketimes = spiketimes[spiketimes > 0]
spiketimes = spiketimes[spiketimes < T_rec]
print(*spiketimes, sep='\n')
