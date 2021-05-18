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

"""Parameters and Settings"""
recorded_system = 'CA1'
rec_length = '90min'
number_valid_neurons = 28

PLOTTING_DIR = dirname(realpath(__file__))
CODE_DIR = '{}/../..'.format(PLOTTING_DIR)
path.insert(1, '{}/src'.format(CODE_DIR))
use_settings_path = False

if 'hde_glm' not in modules:
    import hde_glm as glm
    import hde_utils as utl
    import hde_plotutils as plots

R_bbc_uniform_list = []
R_shuffling_uniform_list = []
R_fivebins_uniform_list = []
"""Load data"""
for neuron_index in range(number_valid_neurons):
    setup = 'full_bbc'
    ANALYSIS_DIR, analysis_num_str, R_tot_bbc, T_D_bbc, T, R_bbc, R_bbc_CI_lo, R_bbc_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = 'bbc', use_settings_path= use_settings_path)

    setup = 'full_bbc_uniform'
    ANALYSIS_DIR, analysis_num_str, R_tot_bbc_uniform, T_D_bbc_uniform, T, R_bbc_uniform, R_bbc_uniform_CI_lo, R_bbc_uniform_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = 'bbc', use_settings_path= use_settings_path)
    R_bbc_uniform = R_bbc_uniform/R_bbc
    R_bbc_uniform_list += [R_bbc_uniform]

    setup = 'full_shuffling'
    ANALYSIS_DIR, analysis_num_str, R_tot_shuffling, T_D_shuffling, T, R_shuffling, R_shuffling_CI_lo, R_shuffling_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = 'shuffling', use_settings_path= use_settings_path)

    setup = 'full_shuffling_uniform'
    ANALYSIS_DIR, analysis_num_str, R_tot_shuffling_uniform, T_D_shuffling_uniform, T, R_shuffling_uniform, R_shuffling_uniform_CI_lo, R_shuffling_uniform_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = 'shuffling', use_settings_path= use_settings_path)
    R_shuffling_uniform = R_shuffling_uniform/R_shuffling
    R_shuffling_uniform_list += [R_shuffling_uniform]

    setup = 'fivebins'
    ANALYSIS_DIR, analysis_num_str, R_tot_fivebins, T_D_fivebins, T, R_fivebins, R_fivebins_CI_lo, R_fivebins_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = 'shuffling',use_settings_path= use_settings_path)

    setup = 'fivebins_uniform'
    ANALYSIS_DIR, analysis_num_str, R_tot_fivebins_uniform, T_D_fivebins_uniform, T, R_fivebins_uniform, R_fivebins_uniform_CI_lo, R_fivebins_uniform_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = 'shuffling',use_settings_path= use_settings_path)
    R_fivebins_uniform = R_fivebins_uniform/R_bbc
    R_fivebins_uniform_list += [R_fivebins_uniform]


R_bbc_uniform_median = np.sort(np.transpose(R_bbc_uniform_list), axis=1)[
    :, int(number_valid_neurons / 2)]
R_bbc_uniform_loPC = np.sort(np.transpose(R_bbc_uniform_list), axis=1)[
    :, int(number_valid_neurons * 0.05)]
R_bbc_uniform_hiPC = np.sort(np.transpose(R_bbc_uniform_list), axis=1)[
    :, int(number_valid_neurons * 0.95)]
R_shuffling_uniform_median = np.sort(np.transpose(R_shuffling_uniform_list), axis=1)[
    :, int(number_valid_neurons / 2)]
R_shuffling_uniform_loPC = np.sort(np.transpose(R_shuffling_uniform_list), axis=1)[
    :, int(number_valid_neurons * 0.05)]
R_shuffling_uniform_hiPC = np.sort(np.transpose(R_shuffling_uniform_list), axis=1)[
    :, int(number_valid_neurons * 0.95)]
R_fivebins_uniform_median = np.sort(np.transpose(R_fivebins_uniform_list), axis=1)[
    :, int(number_valid_neurons / 2)]
R_fivebins_uniform_loPC = np.sort(np.transpose(R_fivebins_uniform_list), axis=1)[
    :, int(number_valid_neurons * 0.05)]
R_fivebins_uniform_hiPC = np.sort(np.transpose(R_fivebins_uniform_list), axis=1)[
    :, int(number_valid_neurons * 0.95)]


"""Plotting"""
# Colors
rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '15.0'
matplotlib.rcParams['xtick.labelsize'] = '15'
matplotlib.rcParams['ytick.labelsize'] = '15'
matplotlib.rcParams['legend.fontsize'] = '15'
matplotlib.rcParams['axes.linewidth'] = 0.6

main_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]
green = sns.cubehelix_palette(8, start=.5, rot=-.75)[3]
soft_red = sns.color_palette("RdBu_r", 15)[10]
soft_blue = sns.color_palette("RdBu_r", 15)[4]


fig, ((ax)) = plt.subplots(1, 1, figsize=(3.5, 2.8))

##### x-axis ####
ax.set_xscale('log')
ax.set_xlim((5, 5000.))
ax.set_xticks(np.array([10, 100, 1000]))
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.1,0.2, 0.3, 0.4, 0.5,0.6,0.7, 0.8, 0.9),numticks=12)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.spines['bottom'].set_bounds(5, 5000)
ax.set_xlabel(r'past range  $T$ (ms)')

##### y-axis ####
ax.set_ylabel(r'\begin{center} $R(T)$ relative to \\ exponential embedding \end{center}')

##### Unset Borders #####
ax.spines['top'].set_bounds(0, 0)
ax.spines['right'].set_bounds(0, 0)

T = T*1000 # time in ms
ax.plot(T, R_bbc_uniform_median,
        linewidth=2,  color=main_red, zorder=1)
ax.fill_between(T, R_bbc_uniform_loPC, R_bbc_uniform_hiPC,
                facecolor=main_red, alpha=0.4)
ax.plot(T, R_shuffling_uniform_median,
        linewidth=2,  color=main_blue, zorder=1)
ax.fill_between(T, R_shuffling_uniform_loPC, R_shuffling_uniform_hiPC,
                facecolor=main_blue, alpha=0.4)

ax.plot(T, R_fivebins_uniform_median,
        linewidth=2, color=green, zorder=1)
ax.fill_between(T, R_fivebins_uniform_loPC, R_fivebins_uniform_hiPC,
                facecolor=green, alpha=0.4)
ax.plot(T, np.zeros(len(T))+1,
        linewidth=2, color='0.0', zorder=2)
ax.plot(T, np.zeros(len(T))+1,
        linewidth=2.5, color=green, zorder=1)
ax.text(80, 0.58, r'\begin{center}uniform \\($\kappa = 0$)\end{center}',
        color='0.0', ha='left', va='bottom')
ax.text(170, 1.04, r'\begin{center}exponential \\($\kappa$ optimized)\end{center}',
        color='0.0', ha='left', va='bottom')

fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

plt.savefig('{}/Fig6C_R_vs_T_relative_to_exponential.pdf'.format(PLOTTING_DIR),
            format="pdf", bbox_inches='tight')
plt.show()
plt.close()
