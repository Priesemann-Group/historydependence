from sys import stderr, exit, argv
from os.path import isfile, isdir, realpath, dirname, exists
import numpy as np

noise_index = int(argv[1])
rec_length = argv[2]

if len(argv) > 3:
    data_path = argv[3]
else:
    CODE_DIR = '{}/..'.format(dirname(realpath(__file__)))
    data_path = '{}/data'.format(CODE_DIR)

DATA_DIR = '{}/izhikevich_neuron'.format(data_path)

# spiketimes = np.load('{}/spiketimes_20min.npy'.format(DATA_DIR))
spiketimes = np.load('%s/spiketimes_%s_noiseindex%d.npy'%(DATA_DIR,rec_length, noise_index))

print(*spiketimes, sep='\n')
