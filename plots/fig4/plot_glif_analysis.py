
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
recorded_system = 'glif_22s_kernel'
rec_length = '900min'
T_0 = 0.00997
T_0_ms = T_0 * 1000
t_0 = 100. - 10**(-4)
bin_size = 0.005
bin_size_ms = int(bin_size*1000)
# add by plus one because T_0 refers to lower bin edge, while the toolbox works with the upper edges of time bins (or steps)
min_step_autocorrelation = 3
max_step_autocorrelation = 600

"""Load and preprocess data"""
DATA_DIR = '{}/data/{}'.format(CODE_DIR, recorded_system)
spiketimes = np.load('{}/spiketimes_{}.npy'.format(DATA_DIR,rec_length))
spiketimes = spiketimes - t_0
spiketimes = spiketimes[spiketimes > 0]
spiketimes = spiketimes - spiketimes[0]
Trec = spiketimes[-1] - spiketimes[0]
counts_from_sptimes = utl.get_binned_neuron_activity(spiketimes, bin_size)

"""Compute measures"""
# Corr
rk = mre.coefficients(counts_from_sptimes, dt=bin_size_ms, steps = (min_step_autocorrelation, max_step_autocorrelation))
T_C_ms = rk.steps*bin_size_ms

# R and Delta R
R_tot = np.load('{}/analysis/{}/R_tot_simulation.npy'.format(CODE_DIR, recorded_system))
T_R, R = plots.load_analysis_results_glm_Simulation(CODE_DIR, recorded_system, use_settings_path = use_settings_path)
dR_arr = plots.get_dR(T_R,R ,R_tot)
# transform T to ms
T_R_ms = T_R * 1000
tau_R = plots.get_T_avg(T_R_ms, dR_arr, T_0_ms)
T_R_plotting = np.append([0], T_R_ms)
R_plotting = np.append([0], R)

# lagged mutual information
lagged_MI = np.load('{}/analysis/{}/lagged_MI.npy'.format(CODE_DIR, recorded_system))
tau_L = plots.get_T_avg(T_R_ms, lagged_MI, T_0_ms)


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
        xmin = 10
        xmax = 3000
        ax.set_xlim((xmin,xmax))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.spines['bottom'].set_bounds(xmin, xmax)
        ax.set_xticks([10, 100, 1000])

fig.text(0.52,  0.503, r'time lag $T$ (ms)', ha='center', va='center', fontsize = 15)
fig.text(0.52,  0.02, r'past range $T$ (ms)', ha='center', va='center', fontsize = 15)

# Plot autocorrelation
ax1.step(np.append([0],T_C_ms[:-1]), rk.coefficients, color = green, where = 'post')
ax1.legend(loc = ((0.00,0.01)), frameon=False)
ax1.set_ylabel(r'$C(T)$')

# Plot lagged mutualinfo (L) and Delta R
ax3.set_ylabel(r'\begin{center} $R(T)$  \end{center}')
ax3.plot(T_R_plotting, R_plotting, color = main_blue)
ax3.text(15, R_tot, r'$R_{\mathrm{tot}}$',
        color='0.0', ha='left', va='bottom')
ax3.axhline(y=R_tot, xmax=1.0,
                    linewidth=1.5, linestyle='--',color = '0.4', zorder = 3)

ax4.set_ylabel(r'gain $\Delta R(T)$')
ax4.axvline(x=tau_R, ymax=1.0, linewidth=1.5, linestyle='--',color = main_blue, zorder = 3)
ax4.text(tau_R*1.3, 0.007, r'$\tau_R$',
        color='0.0', ha='left', va='bottom')
ax4.step(T_R_ms[:-1], dR_arr[1:], color = main_blue, where = 'post')

ax2.step(T_R_ms[:-1], lagged_MI[1:],  color = 'orange', where = 'post')
ax2.axvline(x=tau_L, ymax=1.0, linewidth=1.5, linestyle='--',color = 'orange', zorder = 3)
ax2.text(tau_L*1.3, 0.007, r'$\tau_{L}$',
        color='0.0', ha='left', va='bottom')
ax2.set_ylim(ax4.get_ylim())
ax2.set_ylabel(r'$L(T)/H$')

fig.tight_layout(pad = 1.1, h_pad =1.5, w_pad = 2.0)
# Save fig
plt.savefig('{}/Fig4_glif_analysis.pdf'.format(PLOTTING_DIR, recorded_system),
            format="pdf", bbox_inches='tight')
plt.show()
plt.close()
