
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

PLOTTING_DIR = dirname(realpath(__file__))
CODE_DIR = '{}/../..'.format(PLOTTING_DIR)
path.insert(1, '{}/src'.format(CODE_DIR))
use_settings_path = False

if 'hde_glm' not in modules:
    import hde_glm as glm
    import hde_utils as utl
    import hde_plotutils as plots

recorded_system = 'glif_1s_kernel'
rec_length = '90min'
sample_index = 4
use_settings_path = False
T_0 = 0.0997

"""Load data """
# load estimate of ground truth
R_tot_true = np.load('{}/analysis/{}/R_tot_900min.npy'.format(CODE_DIR, recorded_system))
T_true, R_true = plots.load_analysis_results_glm_Simulation(CODE_DIR, recorded_system, use_settings_path)
T_true = np.append(T_true, [1.0,  3.0])
R_true = np.append(R_true, [R_tot_true, R_tot_true])
dR_true = plots.get_dR(T_true,R_true,R_tot_true)
tau_R_true = plots.get_T_avg(T_true, dR_true, T_0)

# Load settings from yaml file
setup = 'full_bbc'

ANALYSIS_DIR, analysis_num_str, R_tot_bbc, T_D_bbc, T, R_bbc, R_bbc_CI_lo, R_bbc_CI_hi = plots.load_analysis_results(
    recorded_system, rec_length, sample_index, setup, CODE_DIR, regularization_method='bbc', use_settings_path = use_settings_path)

R_tot_bbc, T_D_index_bbc, max_valid_index_bbc = plots.get_R_tot(T, R_bbc, R_bbc_CI_lo)
dR_bbc = plots.get_dR(T,R_bbc,R_tot_bbc)
tau_R_bbc = plots.get_T_avg(T, dR_bbc, T_0)

glm_bbc_csv_file_name = '{}/ANALYSIS{}/glm_benchmark_bbc.csv'.format(
    ANALYSIS_DIR, analysis_num_str)
glm_bbc_pd = pd.read_csv(glm_bbc_csv_file_name)
R_glm_bbc = np.array(glm_bbc_pd['R_GLM'])
T_glm_bbc = np.array(glm_bbc_pd['T'])

setup = 'full_shuffling'

ANALYSIS_DIR, analysis_num_str, R_tot_shuffling, T_D_shuffling, T, R_shuffling, R_shuffling_CI_lo, R_shuffling_CI_hi = plots.load_analysis_results(
    recorded_system, rec_length, sample_index, setup, CODE_DIR, regularization_method='shuffling', use_settings_path = use_settings_path)

R_tot_shuffling, T_D_index_shuffling, max_valid_index_shuffling = plots.get_R_tot(T, R_shuffling, R_shuffling_CI_lo)
dR_shuffling = plots.get_dR(T,R_shuffling,R_tot_shuffling)
tau_R_shuffling = plots.get_T_avg(T, dR_shuffling, T_0)

print(R_tot_bbc, R_tot_true)
glm_shuffling_csv_file_name = '{}/ANALYSIS{}/glm_benchmark_shuffling.csv'.format(
    ANALYSIS_DIR, analysis_num_str)
glm_shuffling_pd = pd.read_csv(glm_shuffling_csv_file_name)
R_glm_shuffling = np.array(glm_shuffling_pd['R_GLM'])
T_glm_shuffling = np.array(glm_shuffling_pd['T'])

"""Plotting"""
rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '15.0'
matplotlib.rcParams['xtick.labelsize'] = '15'
matplotlib.rcParams['ytick.labelsize'] = '15'
matplotlib.rcParams['legend.fontsize'] = '15'
matplotlib.rcParams['axes.linewidth'] = 0.6

fig, ((ax)) = plt.subplots(1, 1, figsize=(3.5, 3))

# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
soft_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]

##########################################
########## Simulated Conventional ########
##########################################

##### x-axis ####
ax.set_xscale('log')
ax.set_xlim((0.15, 3.))
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.spines['bottom'].set_bounds(0.15, 3)
ax.set_xticks([0.2,1.0])
ax.set_xticklabels([r'$200$',r'$1000$'])

##### y-axis ####
ax.set_ylim((0.14, .22))

##### Unset Borders #####
ax.spines['top'].set_bounds(0, 0)
ax.spines['right'].set_bounds(0, 0)

# GLM true max and R vs T
ax.plot([T[0], T[-1]], [R_tot_true, R_tot_true], '--', color='0.5', zorder=1)
ax.plot(T_true, R_true, color='.5', zorder=6, linewidth = 1.2)

# GLM for same embeddings as comparison
ax.plot(T_glm_bbc, R_glm_bbc, '-.', color='.4', alpha=0.8,
        zorder=3, label='true $R(T)$ (BBC)')  # , label='Model'
ax.plot(T_glm_shuffling, R_glm_shuffling, ':', color='.4',
        lw=1.8, alpha=0.8, zorder=2, label=r'true $R(T)$ (Shuffling)')

# Embedding optimized estimates and confidence intervals
ax.plot(T, R_bbc, linewidth=1.2,  color=main_red, zorder=4)
ax.fill_between(T, R_bbc_CI_lo, R_bbc_CI_hi, facecolor=main_red, alpha=0.3)
ax.plot(T, R_shuffling, linewidth=1.2, color=main_blue, zorder=3)
ax.fill_between(T, R_shuffling_CI_lo, R_shuffling_CI_hi,
                facecolor=main_blue, alpha=0.3)

ax.plot(T[T_D_index_bbc:max_valid_index_bbc], np.zeros(max_valid_index_bbc-T_D_index_bbc)+R_tot_bbc, color=main_red, linestyle='--')
ax.plot(T[T_D_index_shuffling:max_valid_index_shuffling], np.zeros(max_valid_index_shuffling-T_D_index_shuffling)+R_tot_shuffling, color=main_blue, linestyle='--')


ax.plot([T_D_bbc], [R_tot_bbc], marker='|', markersize=10, color=main_red,
        zorder=8)
ax.plot([T[max_valid_index_bbc-1]], [R_tot_bbc], marker='|', markersize=10, color=main_red,zorder=8)


ax.plot([T_D_shuffling], [R_tot_shuffling], marker='|', markersize=10, color=main_blue,
        zorder=8)
ax.plot([T[max_valid_index_shuffling-1]], [R_tot_shuffling], marker='|', markersize=10, color=main_blue,zorder=8)

ax.legend(loc=(.05, .83), frameon=False)

fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
plt.savefig('{}/Fig5C_R_vs_T_zoomin.pdf'.format(PLOTTING_DIR),
            format="pdf", bbox_inches='tight')
plt.show()
plt.close()
