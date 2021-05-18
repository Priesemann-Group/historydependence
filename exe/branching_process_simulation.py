import sys
from sys import stderr, exit, argv, path
from os.path import isfile, isdir, realpath, dirname, exists
import numpy as np
import mrestimator as mre
# plotting
import matplotlib
import seaborn.apionly as sns
from matplotlib import rc
import matplotlib.lines as mlines
import pylab as plt

CODE_DIR = '{}/..'.format(dirname(realpath(__file__)))
path.insert(1, '{}/src'.format(CODE_DIR))
if 'hde_utils' not in sys.modules:
    import hde_utils as utl

import matplotlib.pyplot as plt

def binarize(counts):
    counts[np.nonzero(counts)]=1

rec_length = argv[1]

if len(argv) > 2:
    data_path = argv[2]
else:
    CODE_DIR = '{}/..'.format(dirname(realpath(__file__)))
    data_path = '{}/data'.format(CODE_DIR)

DATA_DIR = '{}/branching_process'.format(data_path)

rec_lengths = {'1min': 60., '3min': 180., '5min': 300.,
               '10min': 600., '20min': 1200., '45min': 2700., '90min': 5400.}

T_rec = rec_lengths[rec_length]

t_simbin = 0.004 #4ms simulation timestep
for tau in [70, 85, 100, 120, 150, 198, 300, 500]:
    m = np.exp(-t_simbin*1000/tau)
    print(tau,m)
    N = int(T_rec/t_simbin)
    Nneurons = 100.
    rate = 5. #Hz
    subsampling = 1/Nneurons
    population_activity = Nneurons*rate*t_simbin

    counts_branching_process = mre.simulate_branching(m, a=population_activity, length=N, subp=subsampling, seed = 300)
    activity_branching_process = mre.simulate_branching(m, a=population_activity, length=N, seed = 300)
    spike_noise = np.random.normal(0,0.001,len(np.nonzero(counts_branching_process)[1]))
    spiketimes = np.nonzero(counts_branching_process)[1]*t_simbin + spike_noise
    print(tau, np.sum(activity_branching_process)/N)
    np.save('{}/spiketimes_tau{}ms_{}.npy'.format(DATA_DIR,int(round(tau)),rec_length), spiketimes)
