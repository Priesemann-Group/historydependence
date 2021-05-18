
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
from matplotlib.ticker import NullFormatter

PLOTTING_DIR = dirname(realpath(__file__))
CODE_DIR = '{}/../..'.format(PLOTTING_DIR)
path.insert(1, '{}/src'.format(CODE_DIR))
use_settings_path = False
T_0 = 0.00997

if 'hde_glm' not in modules:
    import hde_glm as glm
    import hde_utils as utl
    import hde_plotutils as plots


def median_relative_mean_R_tot_and_T_avg(recorded_system, setup, N_neurons, rec_lengths, rec_lengths_Nsamples, CODE_DIR):
    if recorded_system == 'CA1':
        DATA_DIR = '{}/data/CA1/'.format(CODE_DIR)
    if recorded_system == 'retina':
        DATA_DIR = '{}/data/retina/'.format(CODE_DIR)
    if recorded_system == 'culture':
        DATA_DIR = '{}/data/culture/'.format(CODE_DIR)
    validNeurons = np.load(
        '{}validNeurons.npy'.format(DATA_DIR)).astype(int)
    R_tot_relative_mean = {}
    T_avg_relative_mean = {}
    np.random.seed(41)
    neuron_selection = np.random.choice(len(validNeurons), N_neurons,  replace=False)
    for rec_length in rec_lengths:
        # arrays containing R_tot and mean T_avg for different neurons
        R_tot_mean_arr = []
        T_avg_mean_arr = []
        N_samples = rec_lengths_Nsamples[rec_length]
        for j in range(N_neurons):
            neuron_index = neuron_selection[j]
            R_tot_arr = []
            T_avg_arr = []
            for sample_index in range(N_samples):
                # Get run index
                run_index = j * N_samples + sample_index
                """Load data five bins"""
                if not rec_length == '90min':
                    setup_subsampled = '{}_subsampled'.format(setup)
                else:
                    run_index = neuron_index
                    setup_subsampled = setup
                if setup == 'full_bbc':
                    analysis_results = plots.load_analysis_results(
                        recorded_system, rec_length, run_index, setup_subsampled, CODE_DIR, regularization_method='bbc', use_settings_path=use_settings_path)
                else:
                    analysis_results = plots.load_analysis_results(
                        recorded_system, rec_length, run_index, setup_subsampled, CODE_DIR, regularization_method='shuffling', use_settings_path=use_settings_path)
                if not analysis_results == None:
                    ANALYSIS_DIR, analysis_num_str, R_tot, T_D, T, R, R_CI_lo, R_CI_hi = analysis_results
                    if not len(R) == 0:
                        R_tot_analysis_results = plots.get_R_tot(T, R, R_CI_lo)
                        if not R_tot_analysis_results == None:
                            R_tot, T_D_index, max_valid_index = R_tot_analysis_results
                            # R_running_avg = plots.get_running_avg(R)
                            dR = plots.get_dR(T,R,R_tot)
                            T_avg = plots.get_T_avg(T, dR, T_0)
                            T_avg_arr += [T_avg]
                            R_tot_arr += [R_tot]
                        else:
                            print('CI_fail', recorded_system, setup, rec_length, run_index, neuron_index, sample_index)
                    else:
                        print('no valid embeddings', recorded_system, rec_length,  setup, analysis_num_str)
                else:
                    print('analysis_fail', recorded_system, rec_length, setup, run_index, neuron_index, sample_index)
            R_tot_mean_arr += [np.mean(R_tot_arr)]
            T_avg_mean_arr += [np.mean(T_avg_arr)]
        R_tot_relative_mean[rec_length] = np.array(R_tot_mean_arr)
        T_avg_relative_mean[rec_length] = np.array(T_avg_mean_arr)

    median_R_tot_relative_mean = []
    median_CI_R_tot_relative_mean = []
    median_T_avg_relative_mean = []
    median_CI_T_avg_relative_mean = []
    for rec_length in rec_lengths:
        R_tot_relative_mean_arr = R_tot_relative_mean[rec_length] / R_tot_relative_mean['90min']*100
        T_avg_relative_mean_arr = T_avg_relative_mean[rec_length] / T_avg_relative_mean['90min']*100
        # If no valid embeddings were found for BBC for all samples, the mean is nan so the neuron is not considered in the median operation
        R_tot_relative_mean_arr = R_tot_relative_mean_arr[~np.isnan(R_tot_relative_mean_arr)]
        T_avg_relative_mean_arr = T_avg_relative_mean_arr[~np.isnan(T_avg_relative_mean_arr)]
        # Computing the median and 95% CIs over the 10 neurons
        median_R_tot_relative_mean += [np.median(R_tot_relative_mean_arr)]
        median_CI_R_tot_relative_mean += [plots.get_CI_median(R_tot_relative_mean_arr)]
        median_T_avg_relative_mean += [np.median(T_avg_relative_mean_arr)]
        median_CI_T_avg_relative_mean += [plots.get_CI_median(T_avg_relative_mean_arr)]
    return np.array(median_R_tot_relative_mean), np.array(median_CI_R_tot_relative_mean), np.array(median_T_avg_relative_mean), np.array(median_CI_T_avg_relative_mean)


N_neurons = 10
# 1min excluded, because just too little data for neurons with low firing rates
rec_lengths = ['3min', '5min', '10min', '20min', '45min', '90min']
rec_length_values = [180., 300., 600., 1200., 2700., 5400.]
rec_lengths_Nsamples = {'3min': 10, '5min': 10,
                        '10min': 8, '20min': 4, '45min': 2, '90min': 1}

"""Plotting"""
rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '15.0'
matplotlib.rcParams['xtick.labelsize'] = '15'
matplotlib.rcParams['ytick.labelsize'] = '15'
matplotlib.rcParams['legend.fontsize'] = '15'
matplotlib.rcParams['axes.linewidth'] = 0.6


# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]
green = sns.cubehelix_palette(8, start=.5, rot=-.75)[3]


"""Load data for all plots"""

"""Plotting"""
fig, (axes) = plt.subplots(3, 2, figsize=(10, 9.5))
for i, recorded_system in enumerate(['CA1', 'retina', 'culture']):
    setup = 'fivebins'
    median_R_tot_relative_mean_fivebins, median_CI_R_tot_relative_mean_fivebins, median_T_avg_relative_mean_fivebins, median_CI_T_avg_relative_mean_fivebins = median_relative_mean_R_tot_and_T_avg(recorded_system, setup, N_neurons, rec_lengths, rec_lengths_Nsamples, CODE_DIR)
    setup = 'full_bbc'
    median_R_tot_relative_mean_bbc, median_CI_R_tot_relative_mean_bbc, median_T_avg_relative_mean_bbc, median_CI_T_avg_relative_mean_bbc = median_relative_mean_R_tot_and_T_avg(recorded_system, setup, N_neurons, rec_lengths, rec_lengths_Nsamples, CODE_DIR)
    setup = 'full_shuffling'
    median_R_tot_relative_mean_shuffling, median_CI_R_tot_relative_mean_shuffling, median_T_avg_relative_mean_shuffling, median_CI_T_avg_relative_mean_shuffling = median_relative_mean_R_tot_and_T_avg(recorded_system, setup, N_neurons, rec_lengths, rec_lengths_Nsamples, CODE_DIR)
    for j in range(2):
        ax = axes[i][j]
        x = rec_length_values
        labels = ['3', '5', '10', '20', '45', '90']

        ax.set_xscale('log')
        ax.set_xlim((170, 22500))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.spines['bottom'].set_bounds(170, 5500)
        ax.minorticks_off()
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel(r'recording time $T_{\mathrm{rec}}$ (min)')

        ##### Unset Borders #####
        ax.spines['top'].set_bounds(0, 0)
        ax.spines['right'].set_bounds(0, 0)
        ax.plot(rec_length_values, np.zeros(len(rec_lengths))+100, linestyle='--', color='0')
        # only plot labels and legend for left-hand side
        if j == 0:
            ax.set_ylim((0, 120.5))
            ax.spines['left'].set_bounds(0, 100)
            if i == 0:
                ax.set_ylabel(r'\begin{center}total history dependence $R_{\mathrm{tot}}$ \\ relative to $90\,\mathrm{min}$ estimate (\%)\end{center}')
        if j == 1:
            ax.set_ylim((0, 120.5))
            ax.spines['left'].set_bounds(0, 100)
            if i == 0:
                ax.set_ylabel(r'\begin{center}information timescale $\tau_{R}$\\ relative to $90\,\mathrm{min}$ estimate (\%)\end{center}')
            else:
                ax.set_title(r'bla bla', alpha=0.0)

        for k in range(len(rec_lengths)):
            if j == 0:
                median_val_fivebins = median_R_tot_relative_mean_fivebins[k]
                median_CI_val_fivebins = median_CI_R_tot_relative_mean_fivebins[k]
                median_val_bbc = median_R_tot_relative_mean_bbc[k]
                median_CI_val_bbc = median_CI_R_tot_relative_mean_bbc[k]
                median_val_shuffling = median_R_tot_relative_mean_shuffling[k]
                median_CI_val_shuffling = median_CI_R_tot_relative_mean_shuffling[k]
            if j == 1:
                median_val_fivebins = median_T_avg_relative_mean_fivebins[k]
                median_CI_val_fivebins = median_CI_T_avg_relative_mean_fivebins[k]
                median_val_bbc = median_T_avg_relative_mean_bbc[k]
                median_CI_val_bbc = median_CI_T_avg_relative_mean_bbc[k]
                median_val_shuffling = median_T_avg_relative_mean_shuffling[k]
                median_CI_val_shuffling = median_CI_T_avg_relative_mean_shuffling[k]

            # Fivebins
            median_CI_hi_fivebins = median_CI_val_fivebins[1] - median_val_fivebins
            median_CI_lo_fivebins = median_CI_val_fivebins[0] - median_val_fivebins
            ax.errorbar(x=rec_length_values[k], y=median_val_fivebins, yerr=[[-median_CI_lo_fivebins], [median_CI_hi_fivebins]],
                        color=green, marker='d', markersize=5., capsize=3.0)
            # BBC full
            median_CI_hi_bbc = median_CI_val_bbc[1] - median_val_bbc
            median_CI_lo_bbc = median_CI_val_bbc[0] - median_val_bbc
            ax.errorbar(x=rec_length_values[k], y=median_val_bbc, yerr=[[-median_CI_lo_bbc], [median_CI_hi_bbc]],
            color=main_red, marker='d', markersize=5., capsize=3.0)
            # Shuffling full
            median_CI_hi_shuffling = median_CI_val_shuffling[1] - median_val_shuffling
            median_CI_lo_shuffling = median_CI_val_shuffling[0] - median_val_shuffling
            ax.errorbar(x=rec_length_values[k], y=median_val_shuffling, yerr=[[-median_CI_lo_shuffling], [median_CI_hi_shuffling]],
            color=main_blue, marker='d', markersize=5., capsize=3.0)

            if i+j+k == 0:
                ax.errorbar(x=rec_length_values[k], y=median_val_bbc, yerr=[[-median_CI_lo_bbc], [median_CI_hi_bbc]],
                            color=main_red, marker='d', markersize=5., capsize=3.0, label=r'BBC, $d_{\mathrm{max}}=20$')
                ax.errorbar(x=rec_length_values[k], y=median_val_shuffling, yerr=[[-median_CI_lo_shuffling], [median_CI_hi_shuffling]],
                            color=main_blue, marker='d', markersize=5., capsize=3.0, label=r'Shuffling, $d_{\mathrm{max}}=20$')
                ax.errorbar(x=rec_length_values[k], y=median_val_fivebins, yerr=[[-median_CI_lo_fivebins], [median_CI_hi_fivebins]],
                            color=green, marker='d', markersize=5., capsize=3.0, label=r'Shuffling, $d_{\mathrm{max}}=5$')

        if i+j == 0:
            ax.legend(loc=(0.1, 0.1), frameon=False)

fig.text(0.5, .99, r'rat dorsal hippocampus (CA1)', ha='center', va='center', fontsize=20)
fig.text(0.5, .66, r'salamander retina', ha='center', va='center', fontsize=20)
fig.text(0.5, .33, r'rat cortical culture', ha='center', va='center', fontsize=20)
fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
#
plt.savefig('{}/S3Fig.pdf'.format(PLOTTING_DIR),
            format="pdf", bbox_inches='tight')
plt.show()
plt.close()
