
"""Functions"""
import os
import sys
from sys import exit, stderr, argv, path, modules
from os.path import isfile, isdir, realpath, dirname, exists
from scipy.optimize import bisect
import csv
import yaml
import numpy as np
import pandas as pd
# plotting
import seaborn.apionly as sns
from scipy.optimize import bisect
from matplotlib import rc
import matplotlib.lines as mlines
import pylab as plt
import matplotlib

PLOTTING_DIR = dirname(realpath(__file__))
CODE_DIR = '{}/../..'.format(PLOTTING_DIR)
path.insert(1, '{}/src'.format(CODE_DIR))
use_settings_path = False

if 'hde_glm' not in modules:
    import hde_glm as glm
    import hde_utils as utl
    import hde_plotutils as plots

recorded_system = "glif_22s_kernel"
# rec_lengths = ['1min', '3min', '5min', '10min', '20min', '45min', '90min']
rec_lengths = ['1min', '3min', '5min', '10min', '20min', '45min', '90min']
sample_index = 0
T_0 = 0.00997

"""Plotting"""
rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '15.0'
matplotlib.rcParams['xtick.labelsize'] = '15'
matplotlib.rcParams['ytick.labelsize'] = '15'
matplotlib.rcParams['legend.fontsize'] = '15'
matplotlib.rcParams['axes.linewidth'] = 0.6

fig, (axes) = plt.subplots(7, 2, figsize=(11.5, 19.5))

# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
soft_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]

"""Load data for all plots"""

for i, ax in enumerate(axes.flatten()):
    setup_index = i % 2
    rec_length_index = int(i / 2)
    rec_length = rec_lengths[rec_length_index]
    if setup_index == 0:
        bbc_setup = 'full_bbc'
        shuffling_setup = 'full_shuffling'
    else:
        bbc_setup = 'full_bbc_withCV'
        shuffling_setup = 'full_shuffling_withCV'

    """Load data"""
    # Load settings from yaml file

    ANALYSIS_DIR, analysis_num_str, R_tot_bbc, T_D_bbc, T_bbc, R_bbc, R_bbc_CI_lo, R_bbc_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, sample_index, bbc_setup, CODE_DIR, regularization_method = 'bbc', use_settings_path= use_settings_path)

    R_tot_bbc, T_D_index_bbc, max_valid_index_bbc = plots.get_R_tot(T_bbc, R_bbc, R_bbc_CI_lo)
    dR_bbc = plots.get_dR(T_bbc,R_bbc,R_tot_bbc)
    tau_R_bbc = plots.get_T_avg(T_bbc, dR_bbc, T_0)

    glm_bbc_csv_file_name = '{}/ANALYSIS{}/glm_benchmark_bbc.csv'.format(
        ANALYSIS_DIR, analysis_num_str)
    glm_bbc_pd = pd.read_csv(glm_bbc_csv_file_name)
    R_glm_bbc = np.array(glm_bbc_pd['R_GLM'])
    T_glm_bbc = np.array(glm_bbc_pd['T'])

    ANALYSIS_DIR, analysis_num_str, R_tot_shuffling, T_D_shuffling, T_shuffling, R_shuffling, R_shuffling_CI_lo, R_shuffling_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, sample_index, shuffling_setup, CODE_DIR, regularization_method = 'shuffling', use_settings_path= use_settings_path)

    R_tot_shuffling, T_D_index_shuffling, max_valid_index_shuffling = plots.get_R_tot(T_shuffling, R_shuffling, R_shuffling_CI_lo)
    dR_shuffling = plots.get_dR(T_shuffling,R_shuffling,R_tot_shuffling)
    tau_R_shuffling = plots.get_T_avg(T_shuffling, dR_shuffling, T_0)

    glm_shuffling_csv_file_name = '{}/ANALYSIS{}/glm_benchmark_shuffling.csv'.format(
        ANALYSIS_DIR, analysis_num_str)
    glm_shuffling_pd = pd.read_csv(glm_shuffling_csv_file_name)
    R_glm_shuffling = np.array(glm_shuffling_pd['R_GLM'])
    T_glm_shuffling = np.array(glm_shuffling_pd['T'])

    ax.set_xscale('log')
    ax.set_xlim((0.01, 3.5))
    x_min = 0.01
    x_max = 3.5
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.spines['bottom'].set_bounds(0.01, 3)
    ax.set_xticks(np.array([0.01, 0.1, 1.0]))
    ax.set_xticklabels(
        [r'$10$', r'$100$', r'$1000$'], rotation='horizontal')

    ##### y-axis ####
    if recorded_system == 'simulation':
        ax.set_ylim((0.0, .16))
        ax.set_yticks([0.0, 0.08, 0.16])
        yrange = 0.16
        ymin = 0.0
        ax.spines['left'].set_bounds(.0, 0.16)
    else:
        ax.set_ylim((0.0, .22))
        ax.set_yticks([0.0, 0.1, 0.2])
        yrange = 0.22
        ymin = 0.0
        ax.spines['left'].set_bounds(.0, 0.22)

    ##### Unset Borders #####
    ax.spines['top'].set_bounds(0, 0)
    ax.spines['right'].set_bounds(0, 0)

    if i ==0:
        if recorded_system == 'simulation':
            ax.text(.015, R_tot_shuffling-0.03, r'$\hat{R}_{\mathrm{tot}}$',
                color='0.0', ha='left', va='bottom', fontsize = 20)
        else:
            ax.text(.015, R_tot_shuffling-0.045, r'$\hat{R}_{\mathrm{tot}}$',
                color='0.0', ha='left', va='bottom', fontsize = 20)
        ax.text(tau_R_bbc + 0.45*tau_R_bbc , 0.003, r'$\hat{\tau}_R$',
                color='0.0', ha='left', va='bottom', fontsize = 20)

    if not int(i/2) == 6:
        xlabels = [item.get_text() for item in ax.get_xticklabels()]
        empty_string_labels = ['']*len(xlabels)
        ax.set_xticklabels(empty_string_labels)
    if not i%2 == 0:
        ylabels = [item.get_text() for item in ax.get_yticklabels()]
        empty_string_labels = ['']*len(ylabels)
        ax.set_yticklabels(empty_string_labels)

    # Embedding optimized estimates and confidence intervals
    ax.plot(T_bbc, R_bbc, linewidth=1.2,  color=main_red, zorder=4, label= 'BBC')
    ax.fill_between(T_bbc, R_bbc_CI_lo, R_bbc_CI_hi, facecolor=main_red, alpha=0.3)
    ax.plot(T_shuffling, R_shuffling, linewidth=1.2, color=main_blue, zorder=3, label= 'Shuffling')
    ax.fill_between(T_shuffling, R_shuffling_CI_lo, R_shuffling_CI_hi,
                    facecolor=main_blue, alpha=0.3)

    # Rtot and Tdepth bbc
    ax.plot([tau_R_bbc], [ymin], marker='d', markersize = 5., color=main_red,
             zorder=8)
    ax.axvline(x=tau_R_bbc, ymax=(R_tot_bbc - ymin) / yrange, color=main_red,
                linewidth=0.5, linestyle='--')

    x = (np.log10(T_D_bbc) - np.log10(x_min)) / \
        (np.log10(x_max) - np.log10(x_min))
    ax.axhline(y=R_tot_bbc, xmax=x, color=main_red,
                linewidth=0.5, linestyle='--')
    ax.plot([x_min], [R_tot_bbc], marker='d',markersize = 5., color=main_red,
             zorder=8)
    ax.plot(T_bbc[T_D_index_bbc:max_valid_index_bbc], np.zeros(max_valid_index_bbc-T_D_index_bbc)+R_tot_bbc, color = main_red, linewidth=1.5, linestyle='--')
    ax.plot([T_D_bbc], [R_tot_bbc], marker='|', markersize=7, color=main_red,
            zorder=8)
    ax.plot([T_bbc[max_valid_index_bbc-1]], [R_tot_bbc], marker='|', markersize=7, color=main_red,zorder=8)

    # Rtot and Tdepth Shuffling
    ax.plot([tau_R_shuffling], [ymin], marker='d', markersize = 5., color=main_blue,
             zorder=8)
    ax.axvline(x=tau_R_shuffling, ymax=(R_tot_shuffling - ymin) / yrange, color=main_blue,
                linewidth=0.5, linestyle='--')

    x = (np.log10(T_D_shuffling) - np.log10(x_min)) / \
        (np.log10(x_max) - np.log10(x_min))
    ax.axhline(y=R_tot_shuffling, xmax=x, color=main_blue,
                linewidth=0.5, linestyle='--')
    ax.plot([x_min], [R_tot_shuffling], marker='d',markersize = 5., color=main_blue,
             zorder=8)
    ax.plot(T_shuffling[T_D_index_shuffling:max_valid_index_shuffling], np.zeros(max_valid_index_shuffling-T_D_index_shuffling)+R_tot_shuffling, color = main_blue, linewidth=1.5, linestyle='--')
    ax.plot([T_D_shuffling], [R_tot_shuffling], marker='|', markersize=7, color=main_blue,
            zorder=8)
    ax.plot([T_shuffling[max_valid_index_shuffling-1]], [R_tot_shuffling], marker='|', markersize=7, color=main_blue,zorder=8)

    # GLM for same embeddings
    ax.plot(T_glm_bbc, R_glm_bbc, '-.', color='.4', alpha=0.8,
            zorder=3, label='true $R(T,d^*,\kappa^*)$ (BBC)')  # , label='Model'
    ax.plot(T_glm_shuffling, R_glm_shuffling, ':', color='.4',
            lw=1.8, alpha=0.8, zorder=2, label=r'true $R(T,d^*,\kappa^*)$ (Shuffling)')

    ax.plot(T_shuffling[T_D_index_shuffling:max_valid_index_shuffling], np.zeros(max_valid_index_shuffling-T_D_index_shuffling)+R_tot_shuffling, color = main_blue, linewidth=1.5, linestyle='--')

    if setup_index == 0:
        ax.set_title('{}, no cross-validation'.format(rec_length))
    else:
        ax.set_title('{}, with cross-validation'.format(rec_length))
    if i ==0:
        ax.legend(loc=(0.6, 0.0), frameon=False)

fig.text(0.5, - 0.01, r'past range $T$ (ms)', ha='center', va='center', fontsize = 20)
fig.text(-0.01, 0.5, r'history dependence $R(T)$', ha='center', va='center', rotation='vertical',  fontsize = 20)


fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
plt.savefig('{}/S5Fig_{}.pdf'.format(PLOTTING_DIR, recorded_system),
            format="pdf", bbox_inches='tight')

plt.show()
plt.close()
