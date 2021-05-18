
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
recorded_system = 'branching_process'
# rec_length = '20min'
rec_length = '90min'
setup = 'full_shuffling'
bin_size = 0.004
bin_size_ms = int(bin_size*1000)
min_step_autocorrelation = 1
max_step_autocorrelation = 500
T_0 = 10
T_0 = 0.01
T_0_ms = T_0 * 1000

"""Load and preprocess data"""
DATA_DIR = '{}/data/{}'.format(CODE_DIR, recorded_system)

"""Compute measures"""
tau_list = [70, 85, 100, 120, 150, 198, 300, 500]
m_list = []
R_tot_list = []
tau_R_list = []
for tau in tau_list:
        if tau == 198:
                sample_index = 0
        else:
                sample_index = tau
        m = np.exp(-bin_size*1000/tau)
        ANALYSIS_DIR, analysis_num_str, R_tot, T_D, T_R, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(recorded_system, rec_length, sample_index, setup, CODE_DIR, regularization_method='shuffling', use_settings_path = use_settings_path)
        R_tot, T_D_index, max_valid_index = plots.get_R_tot(T_R, R, R_CI_lo)
        dR_arr = plots.get_dR(T_R, R ,R_tot)
        # transform T to ms
        T_R_ms = T_R * 1000
        tau_R = plots.get_T_avg(T_R_ms, dR_arr, T_0_ms)
        m_list += [m]
        R_tot_list += [R_tot]
        tau_R_list += [tau_R]


"""Plotting"""
rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '15.0'
matplotlib.rcParams['xtick.labelsize'] = '15'
matplotlib.rcParams['ytick.labelsize'] = '15'
matplotlib.rcParams['legend.fontsize'] = '15'
matplotlib.rcParams['axes.linewidth'] = 0.6

fig, ((ax1,ax2)) = plt.subplots(nrows= 2, ncols = 1 , figsize=(2.8, 3.5), sharex = True)

# fig.set_size_inches(4, 3)

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
for ax in [ax1,ax2]:
        ax.spines['top'].set_bounds(0, 0)
        ax.spines['right'].set_bounds(0, 0)
        # ax.set_xlim((xmin,xmax))
        # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.spines['bottom'].set_bounds(round(m_list[0],3), round(m_list[-1],3))
        ax.set_xticks([round(m_list[0],3), round(m_list[3],3), round(m_list[-1],3)])
        # ax.set_xticklabels([r'$T_0$'], rotation='horizontal')
        # ax.set_yticks([])

ax1.set_ylabel(r'\begin{center}$R_{\mathrm{tot}}$\end{center}')
ax2.set
ax2.set_ylabel(r'\begin{center} $\tau_R$ (ms)\end{center}')
ax2.set_xlabel(r'branching parameter $m$')
# Plot autocorrelation
# ax1.set_ylabel('autocorrelation')

ax1.plot(m_list, R_tot_list ,'.', color = main_blue,  markersize = 6)

# ax2.set_ylabel(r'gain $\Delta R$')
#shift dR_arr_long, because values refer to upper bin edge, while we plot the lower
ax2.plot(m_list, tau_R_list ,'.', color = main_blue,  markersize = 6)


fig.tight_layout(pad = 1.1)
# Save fig
plt.savefig('{}/S11Fig_branching_process_measures_vs_m.pdf'.format(PLOTTING_DIR, recorded_system),
            format="pdf", bbox_inches='tight')
plt.show()
plt.close()
