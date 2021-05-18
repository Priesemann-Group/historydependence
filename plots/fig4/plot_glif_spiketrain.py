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
CODE_DIR = '{}/../..'.format(PLOTTING_DIR)

import matplotlib.pyplot as plt

# N = int(T_rec/t_simbin)
data_path = '{}/data/glif_22s_kernel'.format(CODE_DIR)
spiketimes = np.load('{}/spiketimes_900min.npy'.format(data_path))

"""Plotting"""
rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '15.0'
matplotlib.rcParams['xtick.labelsize'] = '15'
matplotlib.rcParams['ytick.labelsize'] = '15'
matplotlib.rcParams['legend.fontsize'] = '15'
matplotlib.rcParams['axes.linewidth'] = 0.6
fig, ((ax)) = plt.subplots(nrows= 1, ncols = 1 , figsize=(2.8, 1.0), sharex = True)
ax.plot(spiketimes[100:110], np.zeros(10), '|', markersize = 9)
ax.set_xticks([23.5, 24])
fig.tight_layout(pad = 1.1)
plt.savefig('{}/Fig4_glif_spiketrain.pdf'.format(PLOTTING_DIR),
            format="pdf", bbox_inches='tight')
plt.show()
plt.close()
