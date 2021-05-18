
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
        import hde_utils as utl
        import hde_plotutils as plots

"""Parameters"""
recorded_system = 'branching_process'
rec_length = '90min'
sample_index = 0
setup = 'full_shuffling'

tau = 198
bin_size = 0.004
bin_size_ms = int(bin_size*1000)
min_step_autocorrelation = 1
max_step_autocorrelation = 500
T_0 = 10
T_0 = 0.01
T_0_ms = T_0 * 1000
bin_size = 0.004
bin_size_ms = int(bin_size*1000)

"""Load and preprocess data"""
DATA_DIR = '{}/data/{}'.format(CODE_DIR, recorded_system)
spiketimes = np.load('{}/spiketimes_tau{}ms_{}.npy'.format(DATA_DIR,int(tau),rec_length))
# same processing as in the HD toolbox
spiketimes = spiketimes - spiketimes[0]
Trec = spiketimes[-1] - spiketimes[0]
counts_from_sptimes = utl.get_binned_neuron_activity(spiketimes, bin_size)

"""Compute measures"""
# Corr
rk = mre.coefficients(counts_from_sptimes, dt=bin_size_ms, steps = (min_step_autocorrelation, max_step_autocorrelation))
T_C_ms = rk.steps*bin_size_ms
fit = mre.fit(rk, steps = (5,500),fitfunc = mre.f_exponential_offset)
tau_est = fit.tau
rk_offset = fit.popt[2]
# computing integrated timescale on raw data
C_raw = rk.coefficients - rk_offset
tau_C_raw = plots.get_T_avg(T_C_ms, C_raw, T_0_ms)
# computing integrated timescale on fitted curve
C_fit = mre.f_exponential_offset(rk.steps, fit.tau/bin_size_ms, *fit.popt[1:])-rk_offset
tau_C_fit = plots.get_T_avg(T_C_ms, C_fit, T_0_ms)

# R and Delta R
ANALYSIS_DIR, analysis_num_str, R_tot, T_D, T_R, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(
    recorded_system, rec_length, sample_index, setup, CODE_DIR, regularization_method='shuffling', use_settings_path = use_settings_path)
T_R_plotting = np.append([0], T_R)*1000
R_plotting = np.append([0], R)
dR_arr = plots.get_dR(T_R, R ,R_tot)
# transform T to ms
T_R_ms = T_R * 1000
tau_R = plots.get_T_avg(T_R_ms, dR_arr, T_0_ms)

# lagged mutualinformation
T_L = np.arange(1,251)*bin_size
T_L_ms = T_L * 1000
lagged_MI = np.load('{}/analysis/{}/lagged_MI.npy'.format(CODE_DIR, recorded_system))
lagged_MI_offset = np.mean(lagged_MI[150:])
lagged_MI_offset_corrected = lagged_MI - lagged_MI_offset
tau_L = plots.get_T_avg(T_L_ms, lagged_MI_offset_corrected, T_0_ms)

"""Plotting"""
rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '15.0'
matplotlib.rcParams['xtick.labelsize'] = '15'
matplotlib.rcParams['ytick.labelsize'] = '15'
matplotlib.rcParams['legend.fontsize'] = '15'
matplotlib.rcParams['axes.linewidth'] = 0.6

# previous
# fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows= 2, ncols = 2 , figsize=(5.8, 4.2))
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
        # ax.set_xscale('log')
        xmin = 0
        # xmax = 1000
        xmax = 300
        ax.set_xlim((xmin,xmax))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # ax.spines['bottom'].set_bounds(T_0_ms, xmax)
        ax.set_xticks([100, 200, 300])
        # ax.set_xticklabels([r'$T_0$'], rotation='horizontal')
        # ax.set_yticks([])

fig.text(0.52,  0.503, r'time lag $T$ (ms)', ha='center', va='center', fontsize = 15)
fig.text(0.52,  0.02, r'past range $T$ (ms)', ha='center', va='center', fontsize = 15)

# Plot autocorrelation
ax1.step(np.append([0],T_C_ms[:-1]), rk.coefficients, color = green, where = 'post')
ax1.plot(np.append([0],T_C_ms[:-1]), mre.f_exponential_offset(rk.steps, fit.tau/bin_size_ms, *fit.popt[1:]), label= 'exponential fit', color = '0.4', linewidth = 1)

ax1.axvline(x=tau, ymax=1.0,
            linewidth=1.5, linestyle='--',color = green, zorder = 3)
ax1.text(tau+10, rk.coefficients[0]*0.95, r'$\tau_C$',
        color='0.0', ha='left', va='bottom')
ax1.legend(loc = ((0.00,0.001)), frameon=False)
ax1.set_ylabel(r'$C(T)$')

# Plot lagged MI
ax2.step(np.append([0],T_L_ms[:-1]), lagged_MI,  color = 'orange', where = 'post')

ax2.axvline(x=tau_L, ymax=1.0,
            linewidth=1.5, linestyle='--',color = 'orange', zorder = 3)
ax2.text(tau_L+10, lagged_MI[0]*0.95, r'$\tau_{L}$',
        color='0.0', ha='left', va='bottom')
ax2.set_ylabel(r'$L(T)/H$')

# Plot R(T)
ax3.set_ylabel(r'\begin{center} $R(T)$  \end{center}')
ax3.plot(T_R_plotting, R_plotting, color = main_blue)

ax3.text(15, R_tot, r'$R_{\mathrm{tot}}$',
        color='0.0', ha='left', va='bottom')
ax3.axhline(y=R_tot, xmax=1.0,
                    linewidth=1.5, linestyle='--',color = '0.4', zorder = 3)

# Plot Delta R(T)
ax4.set_ylabel(r'gain $\Delta R(T)$')
ax4.axvline(x=tau_R, ymax=1.0, linewidth=1.5, linestyle='--',color = main_blue, zorder = 3)
ax4.text(tau_R+10, dR_arr[0]*0.95, r'$\tau_R$',
        color='0.0', ha='left', va='bottom')
ax4.step(np.append([0],T_R_ms[:-1]), dR_arr, color = main_blue, where = 'post')

fig.tight_layout(pad = 1.1, h_pad =1.5, w_pad = 2.0)
# Save fig
plt.savefig('{}/Fig4_branching_process_analysis.pdf'.format(PLOTTING_DIR, recorded_system),
            format="pdf", bbox_inches='tight')
plt.show()
plt.close()
