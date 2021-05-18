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

"""Load data"""
recorded_system = 'CA1'
rec_length = '90min'
number_valid_neurons = 28

R_tot_shuffling_CA1 = []
R_tot_fivebins_CA1 = []
R_tot_onebin_CA1 = []
R_tot_glm_CA1 = []
for neuron_index in range(number_valid_neurons):
    setup = 'full_bbc'
    ANALYSIS_DIR, analysis_num_str, R_tot_bbc, T_D_bbc, T, R_bbc, R_bbc_CI_lo, R_bbc_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = 'bbc', use_settings_path = use_settings_path)
    R_tot_bbc, T_D_index_bbc, max_valid_index_bbc = plots.get_R_tot(T, R_bbc, R_bbc_CI_lo)

    # Load GLM R_tot
    R_tot_glm = plots.load_analysis_results_glm(ANALYSIS_DIR, analysis_num_str)
    R_tot_glm_CA1 += [R_tot_glm/R_tot_bbc]

    setup = 'full_shuffling'
    ANALYSIS_DIR, analysis_num_str, R_tot_shuffling, T_D_shuffling, T, R_shuffling, R_shuffling_CI_lo, R_shuffling_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR,  regularization_method = 'shuffling', use_settings_path = use_settings_path)
    R_tot_shuffling, T_D_index_shuffling, max_valid_index_shuffling = plots.get_R_tot(T, R_shuffling, R_shuffling_CI_lo)
    R_tot_shuffling_CA1 += [R_tot_shuffling/R_tot_bbc]

    setup = 'fivebins'
    ANALYSIS_DIR, analysis_num_str, R_tot_fivebins, T_D_fivebins, T, R_fivebins, R_fivebins_CI_lo, R_fivebins_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = 'shuffling', use_settings_path = use_settings_path)
    R_tot_fivebins, T_D_index_fivebins, max_valid_index_fivebins = plots.get_R_tot(T, R_fivebins, R_fivebins_CI_lo)
    R_tot_fivebins_CA1 += [R_tot_fivebins/R_tot_bbc]

    setup = 'onebin'
    ANALYSIS_DIR, analysis_num_str, R_tot_onebin, T_D_onebin, T, R_onebin, R_onebin_CI_lo, R_onebin_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = 'shuffling', use_settings_path = use_settings_path)
    R_tot_onebin, T_D_index_onebin, max_valid_index_onebin = plots.get_R_tot(T, R_onebin, R_onebin_CI_lo)
    R_tot_onebin_CA1 += [R_tot_onebin/R_tot_bbc]


R_tot_shuffling_CA1_median = np.median(R_tot_shuffling_CA1)
R_tot_shuffling_CA1_median_loCI, R_tot_shuffling_CA1_median_hiCI = plots.get_CI_median(R_tot_shuffling_CA1)
R_tot_fivebins_CA1_median = np.median(R_tot_fivebins_CA1)
R_tot_fivebins_CA1_median_loCI, R_tot_fivebins_CA1_median_hiCI = plots.get_CI_median(R_tot_fivebins_CA1)
R_tot_onebin_CA1_median = np.median(R_tot_onebin_CA1)
R_tot_onebin_CA1_median_loCI, R_tot_onebin_CA1_median_hiCI = plots.get_CI_median(R_tot_onebin_CA1)
R_tot_glm_CA1_median = np.median(R_tot_glm_CA1)
R_tot_glm_CA1_median_loCI, R_tot_glm_CA1_median_hiCI = plots.get_CI_median(R_tot_glm_CA1)

recorded_system = 'retina'
rec_length = '90min'
number_valid_neurons = 28

R_tot_shuffling_Retina = []
R_tot_fivebins_Retina = []
R_tot_onebin_Retina = []
R_tot_glm_Retina = []
for neuron_index in range(number_valid_neurons):
    setup = 'full_bbc'
    ANALYSIS_DIR, analysis_num_str, R_tot_bbc, T_D_bbc, T, R_bbc, R_bbc_CI_lo, R_bbc_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = 'bbc', use_settings_path = use_settings_path)
    R_tot_bbc, T_D_index_bbc, max_valid_index_bbc = plots.get_R_tot(T, R_bbc, R_bbc_CI_lo)

    # Load GLM R_tot
    R_tot_glm = plots.load_analysis_results_glm(ANALYSIS_DIR, analysis_num_str)
    R_tot_glm_Retina += [R_tot_glm/R_tot_bbc]

    setup = 'full_shuffling'
    ANALYSIS_DIR, analysis_num_str, R_tot_shuffling, T_D_shuffling, T, R_shuffling, R_shuffling_CI_lo, R_shuffling_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR,  regularization_method = 'shuffling', use_settings_path = use_settings_path)
    R_tot_shuffling, T_D_index_shuffling, max_valid_index_shuffling = plots.get_R_tot(T, R_shuffling, R_shuffling_CI_lo)
    R_tot_shuffling_Retina += [R_tot_shuffling/R_tot_bbc]

    setup = 'fivebins'
    ANALYSIS_DIR, analysis_num_str, R_tot_fivebins, T_D_fivebins, T, R_fivebins, R_fivebins_CI_lo, R_fivebins_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = 'shuffling', use_settings_path = use_settings_path)
    R_tot_fivebins, T_D_index_fivebins, max_valid_index_fivebins = plots.get_R_tot(T, R_fivebins, R_fivebins_CI_lo)
    R_tot_fivebins_Retina += [R_tot_fivebins/R_tot_bbc]

    setup = 'onebin'
    ANALYSIS_DIR, analysis_num_str, R_tot_onebin, T_D_onebin, T, R_onebin, R_onebin_CI_lo, R_onebin_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = 'shuffling', use_settings_path = use_settings_path)
    R_tot_onebin, T_D_index_onebin, max_valid_index_onebin = plots.get_R_tot(T, R_onebin, R_onebin_CI_lo)
    R_tot_onebin_Retina += [R_tot_onebin/R_tot_bbc]

R_tot_shuffling_Retina_median = np.median(R_tot_shuffling_Retina)
R_tot_shuffling_Retina_median_loCI, R_tot_shuffling_Retina_median_hiCI = plots.get_CI_median(R_tot_shuffling_Retina)
R_tot_fivebins_Retina_median = np.median(R_tot_fivebins_Retina)
R_tot_fivebins_Retina_median_loCI, R_tot_fivebins_Retina_median_hiCI = plots.get_CI_median(R_tot_fivebins_Retina)
R_tot_onebin_Retina_median = np.median(R_tot_onebin_Retina)
R_tot_onebin_Retina_median_loCI, R_tot_onebin_Retina_median_hiCI = plots.get_CI_median(R_tot_onebin_Retina)
R_tot_glm_Retina_median = np.median(R_tot_glm_Retina)
R_tot_glm_Retina_median_loCI, R_tot_glm_Retina_median_hiCI = plots.get_CI_median(R_tot_glm_Retina)

recorded_system = 'culture'
rec_length = '90min'
number_valid_neurons = 48

R_tot_shuffling_Culture = []
R_tot_fivebins_Culture = []
R_tot_onebin_Culture = []
R_tot_glm_Culture = []
for neuron_index in range(number_valid_neurons):
    setup = 'full_bbc'
    ANALYSIS_DIR, analysis_num_str, R_tot_bbc, T_D_bbc, T, R_bbc, R_bbc_CI_lo, R_bbc_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = 'bbc', use_settings_path = use_settings_path)
    R_tot_bbc, T_D_index_bbc, max_valid_index_bbc = plots.get_R_tot(T, R_bbc, R_bbc_CI_lo)

    # Load GLM R_tot
    R_tot_glm = plots.load_analysis_results_glm(ANALYSIS_DIR, analysis_num_str)
    R_tot_glm_Culture += [R_tot_glm/R_tot_bbc]

    setup = 'full_shuffling'
    ANALYSIS_DIR, analysis_num_str, R_tot_shuffling, T_D_shuffling, T, R_shuffling, R_shuffling_CI_lo, R_shuffling_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR,  regularization_method = 'shuffling', use_settings_path = use_settings_path)
    R_tot_shuffling, T_D_index_shuffling, max_valid_index_shuffling = plots.get_R_tot(T, R_shuffling, R_shuffling_CI_lo)
    R_tot_shuffling_Culture += [R_tot_shuffling/R_tot_bbc]

    setup = 'fivebins'
    ANALYSIS_DIR, analysis_num_str, R_tot_fivebins, T_D_fivebins, T, R_fivebins, R_fivebins_CI_lo, R_fivebins_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = 'shuffling', use_settings_path = use_settings_path)
    R_tot_fivebins, T_D_index_fivebins, max_valid_index_fivebins = plots.get_R_tot(T, R_fivebins, R_fivebins_CI_lo)
    R_tot_fivebins_Culture += [R_tot_fivebins/R_tot_bbc]

    setup = 'onebin'
    ANALYSIS_DIR, analysis_num_str, R_tot_onebin, T_D_onebin, T, R_onebin, R_onebin_CI_lo, R_onebin_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = 'shuffling', use_settings_path = use_settings_path)
    R_tot_onebin, T_D_index_onebin, max_valid_index_onebin = plots.get_R_tot(T, R_onebin, R_onebin_CI_lo)
    R_tot_onebin_Culture += [R_tot_onebin/R_tot_bbc]

R_tot_shuffling_Culture_median = np.median(R_tot_shuffling_Culture)
R_tot_shuffling_Culture_median_loCI, R_tot_shuffling_Culture_median_hiCI = plots.get_CI_median(R_tot_shuffling_Culture)
R_tot_fivebins_Culture_median = np.median(R_tot_fivebins_Culture)
R_tot_fivebins_Culture_median_loCI, R_tot_fivebins_Culture_median_hiCI = plots.get_CI_median(R_tot_fivebins_Culture)
R_tot_onebin_Culture_median = np.median(R_tot_onebin_Culture)
R_tot_onebin_Culture_median_loCI, R_tot_onebin_Culture_median_hiCI = plots.get_CI_median(R_tot_onebin_Culture)
R_tot_glm_Culture_median = np.median(R_tot_glm_Culture)
R_tot_glm_Culture_median_loCI, R_tot_glm_Culture_median_hiCI = plots.get_CI_median(R_tot_glm_Culture)

# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]
soft_red = sns.color_palette("RdBu_r", 15)[10]
soft_blue = sns.color_palette("RdBu_r", 15)[4]
violet = sns.cubehelix_palette(8)[4]
green = sns.cubehelix_palette(8, start=.5, rot=-.75)[3]


fig = plt.figure(figsize=(5., 3.))

ax = plt.subplot2grid((17, 1), (0, 0), colspan=1, rowspan=15)
ax2 = plt.subplot2grid((17, 1), (16, 0), colspan=1, rowspan=1, sharex=ax)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.set_ylabel(
    r'\begin{center} total history dependence \\ $\hat{R}_{\mathrm{tot}}$ relative to BBC \end{center}')
ax.set_ylim((0.55, 1.1))
ax.set_yticks([0.6, 0.8, 1.0])
ax.spines['left'].set_bounds(.55, 1.)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)  # labels along the bottom edge are off

ax2.set_ylim((0.0, 0.05))
ax2.set_yticks([0.0])
ax2.spines['left'].set_bounds(0, 0.05)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.set_xticklabels(
    ['CA1', 'retina', 'culture'], rotation='horizontal')
# ax2.set_xticks(np.array([1, 10, 50]))
x = [1., 4., 7.]
ax2.set_xlim((-.5, 8.5))
ax2.spines['bottom'].set_bounds(-.5, 8.5)
ax2.set_xticks(x)

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
# ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
# ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

fig.add_subplot(ax)
fig.add_subplot(ax2)
fig.subplots_adjust(hspace=0.1)


rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '15.0'
matplotlib.rcParams['xtick.labelsize'] = '15'
matplotlib.rcParams['ytick.labelsize'] = '15'
matplotlib.rcParams['legend.fontsize'] = '15'
matplotlib.rcParams['axes.linewidth'] = 0.6

# EC
ax.bar(x=[0.0], height=[1], yerr=[[0], [0]], width=.5, alpha=.95,
       color=main_red, ecolor="0.3")
ax.bar(x=[0.5], height=[R_tot_shuffling_CA1_median], yerr=[[R_tot_shuffling_CA1_median-R_tot_shuffling_CA1_median_loCI], [R_tot_shuffling_CA1_median_hiCI-R_tot_shuffling_CA1_median]], width=.5, alpha=.95, color=main_blue, ecolor="0.3")
ax.bar(x=[1.0], height=[R_tot_fivebins_CA1_median], yerr=[[R_tot_fivebins_CA1_median-R_tot_fivebins_CA1_median_loCI], [R_tot_fivebins_CA1_median_hiCI-R_tot_fivebins_CA1_median]], width=.5, alpha=.95, color=green, ecolor="0.3")
ax.bar(x=[1.5], height=[R_tot_onebin_CA1_median], yerr=[[R_tot_onebin_CA1_median-R_tot_onebin_CA1_median_loCI], [R_tot_onebin_CA1_median_hiCI-R_tot_onebin_CA1_median]], width=.5, alpha=.95,color='y', ecolor="0.3")
ax.bar(x=[2.0], height=[R_tot_glm_CA1_median], yerr=[[R_tot_glm_CA1_median-R_tot_glm_CA1_median_loCI], [R_tot_glm_CA1_median_hiCI-R_tot_glm_CA1_median]], width=.5, alpha=.95,color=violet, ecolor="0.3")

# Retina
ax.bar(x=[3.0], height=[1], yerr=[[0], [0]], width=.5, alpha=.95,
       color=main_red, ecolor="0.3")
ax.bar(x=[3.5], height=[R_tot_shuffling_Retina_median], yerr=[[R_tot_shuffling_Retina_median-R_tot_shuffling_Retina_median_loCI], [R_tot_shuffling_Retina_median_hiCI - R_tot_shuffling_Retina_median]], width=.5, alpha=.95, color=main_blue, ecolor="0.3")
ax.bar(x=[4.0], height=[R_tot_fivebins_Retina_median], yerr=[[R_tot_fivebins_Retina_median-R_tot_fivebins_Retina_median_loCI], [R_tot_fivebins_Retina_median_hiCI -R_tot_fivebins_Retina_median]], width=.5, alpha=.95, color=green, ecolor="0.3")
ax.bar(x=[4.5], height=[R_tot_onebin_Retina_median], yerr=[[R_tot_onebin_Retina_median-R_tot_onebin_Retina_median_loCI], [R_tot_onebin_Retina_median_hiCI-R_tot_onebin_Retina_median]], width=.5, alpha=.95,color='y', ecolor="0.3")
ax.bar(x=[5.0], height=[R_tot_glm_Retina_median], yerr=[[R_tot_glm_Retina_median-R_tot_glm_Retina_median_loCI], [R_tot_glm_Retina_median_hiCI-R_tot_glm_Retina_median]], width=.5, alpha=.95,color=violet, ecolor="0.3")

# Culture
ax.bar(x=[6.0], height=[1], yerr=[[0], [0]], width=.5, alpha=.95,
       color=main_red, ecolor="0.3", label='BBC')
ax.bar(x=[6.5], height=[R_tot_shuffling_Culture_median], yerr=[[R_tot_shuffling_Culture_median-R_tot_shuffling_Culture_median_loCI], [R_tot_shuffling_Culture_median_hiCI-R_tot_shuffling_Culture_median]], width=.5, alpha=.95,
       color=main_blue, ecolor="0.3", label='Shuffling')
ax.bar(x=[7.0], height=[R_tot_fivebins_Culture_median], yerr=[[R_tot_fivebins_Culture_median-R_tot_fivebins_Culture_median_loCI], [R_tot_fivebins_Culture_median_hiCI-R_tot_fivebins_Culture_median]], width=.5, alpha=.95, color=green, ecolor="0.3", label='max five bins')
ax.bar(x=[7.5], height=[R_tot_onebin_Culture_median], yerr=[[R_tot_onebin_Culture_median-R_tot_onebin_Culture_median_loCI], [R_tot_onebin_Culture_median_hiCI-R_tot_onebin_Culture_median]], width=.5, alpha=.95,color='y', ecolor="0.3", label="single bin")
ax.bar(x=[8.0], height=[R_tot_glm_Culture_median], yerr=[[R_tot_glm_Culture_median-R_tot_glm_Culture_median_loCI], [R_tot_glm_Culture_median_hiCI-R_tot_glm_Culture_median]], width=.5, alpha=.95,color=violet, ecolor="0.3", label='GLM')

number_valid_neurons = 28
ax.scatter(np.zeros(number_valid_neurons) + 0.6, R_tot_shuffling_CA1,
           s=3, color="0.7", marker="o", zorder=9)
ax.scatter(np.zeros(number_valid_neurons) + 1.1, R_tot_fivebins_CA1,
           s=3, color="0.7", marker="o", zorder=2)
ax.scatter(np.zeros(number_valid_neurons) + 1.6, R_tot_onebin_CA1,
           s=3, color="0.7", marker="o", zorder=2)
ax.scatter(np.zeros(number_valid_neurons) + 2.1, R_tot_glm_CA1,
           s=3, color="0.7", marker="o", zorder=2)

number_valid_neurons = 28
ax.scatter(np.zeros(number_valid_neurons) + 3.6, R_tot_shuffling_Retina,
           s=3, color="0.7", marker="o", zorder=9)
ax.scatter(np.zeros(number_valid_neurons) + 4.1, R_tot_fivebins_Retina,
           s=3, color="0.7", marker="o", zorder=2)
ax.scatter(np.zeros(number_valid_neurons) + 4.6, R_tot_onebin_Retina,
           s=3, color="0.7", marker="o", zorder=2)
ax.scatter(np.zeros(number_valid_neurons) + 5.1, R_tot_glm_Retina,
           s=3, color="0.7", marker="o", zorder=2)


number_valid_neurons = 48
ax.scatter(np.zeros(number_valid_neurons) + 6.6, R_tot_shuffling_Culture,
           s=3, color="0.7", marker="o", zorder=9)
ax.scatter(np.zeros(number_valid_neurons) + 7.1, R_tot_fivebins_Culture,
           s=3, color="0.7", marker="o", zorder=2)
ax.scatter(np.zeros(number_valid_neurons) + 7.6, R_tot_onebin_Culture,
           s=3, color="0.7", marker="o", zorder=2)
ax.scatter(np.zeros(number_valid_neurons) + 8.1, R_tot_glm_Culture,
           s=3, color="0.7", marker="o", zorder=2)


ax.axhline(y=1, xmax=8.5, color='0.7',
           linewidth=0.5, linestyle='--')
ax.axhline(y=.95, xmax=8.5, color='0.7',
           linewidth=0.5, linestyle='--')
ax.axhline(y=.85, xmax=8.5, color='0.7',
           linewidth=0.5, linestyle='--')
ax.axhline(y=.90, xmax=8.5, color='0.7',
           linewidth=0.5, linestyle='--')
ax.axhline(y=.80, xmax=8.5, color='0.7',
           linewidth=0.5, linestyle='--')

# EC
ax2.bar(x=[0.0], height=[1], yerr=[[0], [0]], width=.5, alpha=.95,
       color=main_red, ecolor="0.3")
ax2.bar(x=[0.5], height=[R_tot_shuffling_CA1_median], yerr=[[R_tot_shuffling_CA1_median-R_tot_shuffling_CA1_median_loCI], [R_tot_shuffling_CA1_median_hiCI-R_tot_shuffling_CA1_median]], width=.5, alpha=.95, color=main_blue, ecolor="0.3")
ax2.bar(x=[1.0], height=[R_tot_fivebins_CA1_median], yerr=[[R_tot_fivebins_CA1_median-R_tot_fivebins_CA1_median_loCI], [R_tot_fivebins_CA1_median_hiCI-R_tot_fivebins_CA1_median]], width=.5, alpha=.95, color=green, ecolor="0.3")
ax2.bar(x=[1.5], height=[R_tot_onebin_CA1_median], yerr=[[R_tot_onebin_CA1_median-R_tot_onebin_CA1_median_loCI], [R_tot_onebin_CA1_median_hiCI-R_tot_onebin_CA1_median]], width=.5, alpha=.95,color='y', ecolor="0.3")
ax2.bar(x=[2.0], height=[R_tot_glm_CA1_median], yerr=[[R_tot_glm_CA1_median-R_tot_glm_CA1_median_loCI], [R_tot_glm_CA1_median_hiCI-R_tot_glm_CA1_median]], width=.5, alpha=.95,color=violet, ecolor="0.3")

# Retina
ax2.bar(x=[3.0], height=[1], yerr=[[0], [0]], width=.5, alpha=.95,
       color=main_red, ecolor="0.3")
ax2.bar(x=[3.5], height=[R_tot_shuffling_Retina_median], yerr=[[R_tot_shuffling_Retina_median-R_tot_shuffling_Retina_median_loCI], [R_tot_shuffling_Retina_median_hiCI - R_tot_shuffling_Retina_median]], width=.5, alpha=.95, color=main_blue, ecolor="0.3")
ax2.bar(x=[4.0], height=[R_tot_fivebins_Retina_median], yerr=[[R_tot_fivebins_Retina_median-R_tot_fivebins_Retina_median_loCI], [R_tot_fivebins_Retina_median_hiCI -R_tot_fivebins_Retina_median]], width=.5, alpha=.95, color=green, ecolor="0.3")
ax2.bar(x=[4.5], height=[R_tot_onebin_Retina_median], yerr=[[R_tot_onebin_Retina_median-R_tot_onebin_Retina_median_loCI], [R_tot_onebin_Retina_median_hiCI-R_tot_onebin_Retina_median]], width=.5, alpha=.95,color='y', ecolor="0.3")
ax2.bar(x=[5.0], height=[R_tot_glm_Retina_median], yerr=[[R_tot_glm_Retina_median-R_tot_glm_Retina_median_loCI], [R_tot_glm_Retina_median_hiCI-R_tot_glm_Retina_median]], width=.5, alpha=.95,color=violet, ecolor="0.3")

# Culture
ax2.bar(x=[6.0], height=[1], yerr=[[0], [0]], width=.5, alpha=.95,
       color=main_red, ecolor="0.3", label='BBC')
ax2.bar(x=[6.5], height=[R_tot_shuffling_Culture_median], yerr=[[R_tot_shuffling_Culture_median-R_tot_shuffling_Culture_median_loCI], [R_tot_shuffling_Culture_median_hiCI-R_tot_shuffling_Culture_median]], width=.5, alpha=.95,
       color=main_blue, ecolor="0.3", label='Shuffling')
ax2.bar(x=[7.0], height=[R_tot_fivebins_Culture_median], yerr=[[R_tot_fivebins_Culture_median-R_tot_fivebins_Culture_median_loCI], [R_tot_fivebins_Culture_median_hiCI-R_tot_fivebins_Culture_median]], width=.5, alpha=.95, color=green, ecolor="0.3", label='max five bins')
ax2.bar(x=[7.5], height=[R_tot_onebin_Culture_median], yerr=[[R_tot_onebin_Culture_median-R_tot_onebin_Culture_median_loCI], [R_tot_onebin_Culture_median_hiCI-R_tot_onebin_Culture_median]], width=.5, alpha=.95,color='y', ecolor="0.3", label="single bin")
ax2.bar(x=[8.0], height=[R_tot_glm_Culture_median], yerr=[[R_tot_glm_Culture_median-R_tot_glm_Culture_median_loCI], [R_tot_glm_Culture_median_hiCI-R_tot_glm_Culture_median]], width=.5, alpha=.95,color=violet, ecolor="0.3", label='GLM')

plt.savefig('{}/Fig6D_Rtot_relative_to_bbc.pdf'.format(PLOTTING_DIR),
            format="pdf", bbox_inches='tight')

plt.show()
plt.close()
