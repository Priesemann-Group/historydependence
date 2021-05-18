
"""Functions"""
import os
import sys
from sys import exit, stderr, argv, path, modules
from os.path import isfile, isdir, realpath, dirname, exists
import numpy as np
import pandas as pd
# plotting
import matplotlib
import seaborn.apionly as sns
from matplotlib import rc
import matplotlib.lines as mlines
import pylab as plt
import mrestimator as mre

PLOTTING_DIR = dirname(realpath(__file__))
CODE_DIR = '{}/../..'.format(PLOTTING_DIR)
path.insert(1, '{}/src'.format(CODE_DIR))
use_settings_path = False

if 'hde_glm' not in modules:
        import hde_glm as glm
        import hde_utils as utl
        import hde_plotutils as plots

"""Parameters"""
recorded_system = 'izhikevich_neuron'
rec_length = '20min'
sample_index = 0
setup = 'full_bbc'
T_0 = 0.0
T_0_ms = T_0 * 1000
bin_size = 0.001
bin_size_ms = int(bin_size*1000)
# add by plus one because T_0 refers to lower bin edge, while the toolbox works with the upper edges of time bins (or steps)
min_step_autocorrelation = 1
max_step_autocorrelation = 300
N_steps = int(1.0 / 0.005)

"""Load and preprocess data"""
DATA_DIR = '{}/data/{}'.format(CODE_DIR, recorded_system)
spiketimes = np.load('{}/spiketimes_{}.npy'.format(DATA_DIR,rec_length))
spiketimes = spiketimes - spiketimes[0]
Trec = spiketimes[-1] - spiketimes[0]
counts_from_sptimes = utl.get_binned_neuron_activity(spiketimes, bin_size)

"""Compute measures"""
# Corr
rk = mre.coefficients(counts_from_sptimes, dt=bin_size_ms, steps = (min_step_autocorrelation, max_step_autocorrelation))
T_C_ms = rk.steps*bin_size_ms

ANALYSIS_DIR, analysis_num_str, R_tot, T_D, T_R, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(recorded_system, rec_length, sample_index, setup, CODE_DIR, regularization_method='bbc', use_settings_path = use_settings_path)
R_tot, T_D_index, max_valid_index = plots.get_R_tot(T_R, R, R_CI_lo)
dR_arr = plots.get_dR(T_R, R ,R_tot)
T_R_plotting = np.append([0], T_R)*1000
R_plotting = np.append([0], R)
# transform T to ms
T_R_ms = T_R * 1000
tau_R = plots.get_T_avg(T_R_ms, dR_arr, T_0_ms)

# lagged mutual information
T_L = np.append(T_R, np.arange(76,300)*0.001)
T_L_ms = T_L * 1000
lagged_MI = np.load('{}/analysis/{}/lagged_MI.npy'.format(CODE_DIR, recorded_system))
tau_L = plots.get_T_avg(T_L_ms, lagged_MI, T_0_ms)

"""Plotting"""
rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '15.0'
matplotlib.rcParams['xtick.labelsize'] = '15'
matplotlib.rcParams['ytick.labelsize'] = '15'
matplotlib.rcParams['legend.fontsize'] = '15'
matplotlib.rcParams['axes.linewidth'] = 0.6

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows= 2, ncols = 2 , figsize=(5.8, 3.7))

# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]
soft_red = sns.color_palette("RdBu_r", 15)[10]
soft_blue = sns.color_palette("RdBu_r", 15)[4]
violet = sns.cubehelix_palette(8)[4]
green = sns.cubehelix_palette(8, start=.5, rot=-.75)[3]

##########################################
########## Simulated Conventional ########
##########################################

##### x-axis ####
# unset borders
for ax in [ax1,ax2, ax3, ax4]:
        ax.spines['top'].set_bounds(0, 0)
        ax.spines['right'].set_bounds(0, 0)
        ax.set_xscale('log')
        xmin = 1.0
        xmax = 300
        ax.set_xlim((xmin,xmax))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xticks([1, 10, 100])



fig.text(0.52,  0.503, r'time lag $T$ (ms)', ha='center', va='center', fontsize = 15)
fig.text(0.52,  0.02, r'past range $T$ (ms)', ha='center', va='center', fontsize = 15)

# Plot autocorrelation
ax1.step(np.append([0],T_C_ms[:-1]), rk.coefficients, color = green, where = 'post')
ax1.legend(loc = ((0.00,0.01)), frameon=False)
ax1.set_ylabel(r'$C(T)$')

# Plot lagged mutualinfo (L) and Delta R
ax2.step(np.append([0],T_L_ms[:-1]), lagged_MI,  color = 'orange', where = 'post')
ax2.set_ylabel(r'$L(T)/H$')

ax3.set_ylabel(r'\begin{center} $R(T)$  \end{center}')
ax3.plot(T_R_plotting, R_plotting, color = main_blue)
ax3.text(2, R_tot, r'$R_{\mathrm{tot}}$',
        color='0.0', ha='left', va='bottom')
ax3.axhline(y=R_tot, xmax=1.0,
                    linewidth=1.5, linestyle='--',color = '0.4', zorder = 3)

ax4.set_ylabel(r'gain $\Delta R(T)$')
ax4.axvline(x=tau_R, ymax=1.0, linewidth=1.5, linestyle='--',color = main_blue, zorder = 3)
ax4.text(tau_R+1, dR_arr[1]*.95, r'$\tau_R$',
        color='0.0', ha='left', va='bottom')
ax4.step(np.append([0],T_R_ms[:-1]), dR_arr, color = main_blue, where = 'post')
fig.tight_layout(pad = 1.1, h_pad =1.5, w_pad = 2.0)

# Save fig
plt.savefig('{}/Fig4_izhikevich_analysis.pdf'.format(PLOTTING_DIR, recorded_system),
            format="pdf", bbox_inches='tight')
plt.show()
plt.close()
