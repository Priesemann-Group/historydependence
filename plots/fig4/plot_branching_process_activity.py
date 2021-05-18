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

PLOTTING_DIR = dirname(realpath(__file__))

import matplotlib.pyplot as plt

def binarize(counts):
    counts[np.nonzero(counts)]=1

t_simbin = 0.004 #4ms simulation timestep
tau = 198
m = np.exp(-t_simbin*1000/tau)
print(tau,m)
N = 4000
# N = int(T_rec/t_simbin)
Nneurons = 100.
rate = 5. #Hz
subsampling = 1/Nneurons
population_activity = Nneurons*rate*t_simbin

counts_branching_process = mre.simulate_branching(m, a=population_activity, length=N, subp=subsampling, seed = 300)
activity_branching_process = mre.simulate_branching(m, a=population_activity, length=N, seed = 300)

"""Plotting"""
rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '15.0'
matplotlib.rcParams['xtick.labelsize'] = '15'
matplotlib.rcParams['ytick.labelsize'] = '15'
matplotlib.rcParams['legend.fontsize'] = '15'
matplotlib.rcParams['axes.linewidth'] = 0.6
fig, ((ax1,ax2)) = plt.subplots(nrows= 2, ncols = 1 , figsize=(2.8, 3.8), sharex = True)
ax1.set_ylabel('population activity')
counts_branching_process[np.nonzero(counts_branching_process)]=1
ax1.plot(np.arange(2900,3250), activity_branching_process[0][2800:3150])
ax2.plot(np.arange(2900,3250), counts_branching_process[0][2800:3150], '|')
fig.tight_layout(pad = 1.1)
plt.savefig('{}/Fig4_branching_process_activity.pdf'.format(PLOTTING_DIR),
            format="pdf", bbox_inches='tight')
plt.show()
plt.close()
