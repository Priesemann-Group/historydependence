import sys
from sys import stderr, exit, argv, path
from os.path import isfile, isdir, realpath, dirname, exists
import numpy as np

CODE_DIR = '{}/..'.format(dirname(realpath(__file__)))
path.insert(1, '{}/src'.format(CODE_DIR))
if 'hde_utils' not in sys.modules:
    import hde_utils as utl

import matplotlib.pyplot as plt

if int(argv[1])==0:
    tau = 198
else:
    tau = int(argv[1])
rec_length = argv[2]

if len(argv) > 3:
    data_path = argv[3]
else:
    CODE_DIR = '{}/..'.format(dirname(realpath(__file__)))
    data_path = '{}/data'.format(CODE_DIR)

DATA_DIR = '{}/branching_process'.format(data_path)

rec_lengths = {'1min': 60., '3min': 180., '5min': 300.,
               '10min': 600., '20min': 1200., '45min': 2700., '90min': 5400.}

T_rec = rec_lengths[rec_length]

spiketimes = np.load('{}/spiketimes_tau{}ms_90min.npy'.format(DATA_DIR,int(tau)))
spiketimes = spiketimes[spiketimes < T_rec]

print(*spiketimes, sep='\n')
