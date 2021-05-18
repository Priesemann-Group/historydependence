from sys import stderr, exit, argv
from os.path import isfile, isdir, realpath, dirname, exists
import numpy as np
from scipy.io import loadmat
# Loading spiketimes for entorhinal cortex recording

recorded_system = argv[1]
run_index = int(argv[2])
rec_length = argv[3]

if len(argv) > 4:
    data_path = argv[4]
else:
    CODE_DIR = '{}/..'.format(dirname(realpath(__file__)))
    data_path = '{}/data'.format(CODE_DIR)

rec_lengths = {'1min': 60., '3min': 180., '5min': 300.,
               '10min': 600., '20min': 1200., '45min': 2700., '90min': 5400.}

rec_lengths_Nsamples = {'1min': 10, '3min': 10, '5min': 10,
               '10min': 8, '20min': 4, '45min': 2}

DATA_DIR = '{}/{}'.format(data_path, recorded_system)

N_neurons = 10
N_samples = rec_lengths_Nsamples[rec_length]
T_rec = rec_lengths[rec_length]

neuron_index = int(run_index/N_samples)
sample_index = run_index%N_samples

validNeurons = np.load(
    '{}/validNeurons.npy'.format(DATA_DIR)).astype(int)

np.random.seed(41)
neuron_selection = np.random.choice(len(validNeurons), N_neurons,  replace = False)
neuron = validNeurons[neuron_selection][neuron_index]

spiketimes = np.load(
    '{}/spks/spiketimes_neuron{}.npy'.format(DATA_DIR, neuron))

# Add 5 seconds to make sure that only spikes with sufficient spiking history are considered
T_0 = spiketimes[0] + 5.
# End of the recordings seem to be unstable from time to time, therefore only subsample the first 80 minutes
T_step = 4800. / N_samples

T_0 = T_0 + sample_index * T_step

spiketimes = spiketimes - T_0
spiketimes = spiketimes[spiketimes > 0]
spiketimes = spiketimes[spiketimes < T_rec]
print(*spiketimes, sep='\n')
