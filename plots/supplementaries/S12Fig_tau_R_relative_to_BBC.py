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

def get_tau_R(T_0, setup, regularization_method, recorded_system, rec_length, neuron_index, CODE_DIR, use_settings_path):
    ANALYSIS_DIR, analysis_num_str, R_tot, T_D, T, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = regularization_method, use_settings_path = use_settings_path)
    R_tot = plots.get_R_tot(T, R, R_CI_lo)[0]
    dR = plots.get_dR(T,R,R_tot)
    tau_R = plots.get_T_avg(T, dR, T_0)
    return tau_R

def get_tau_R_for_different_estimators(T_0, recorded_system, rec_length, neuron_index, CODE_DIR, use_settings_path):
    setup = 'full_bbc'
    regularization_method = 'bbc'
    tau_R_bbc = get_tau_R(T_0, setup, regularization_method, recorded_system, rec_length, neuron_index, CODE_DIR, use_settings_path)

    setup = 'full_shuffling'
    regularization_method = 'shuffling'
    tau_R_shuffling = get_tau_R(T_0, setup, regularization_method, recorded_system, rec_length, neuron_index, CODE_DIR, use_settings_path)

    setup = 'fivebins'
    regularization_method = 'shuffling'
    tau_R_fivebins = get_tau_R(T_0, setup, regularization_method, recorded_system, rec_length, neuron_index, CODE_DIR, use_settings_path)

    setup = 'onebin'
    regularization_method = 'shuffling'
    tau_R_onebin = get_tau_R(T_0, setup, regularization_method, recorded_system, rec_length, neuron_index, CODE_DIR, use_settings_path)

    return tau_R_bbc, tau_R_shuffling, tau_R_fivebins, tau_R_onebin

"""Parameters"""
rec_length = '90min'
T_0_ms = 10
T_0 = 0.00997

"""Load data"""
recorded_system = 'CA1'
number_valid_neurons = 28

tau_R_shuffling_CA1 = []
tau_R_fivebins_CA1 = []
tau_R_onebin_CA1 = []
for neuron_index in range(number_valid_neurons):
    tau_R_bbc, tau_R_shuffling, tau_R_fivebins, tau_R_onebin = get_tau_R_for_different_estimators(T_0, recorded_system, rec_length, neuron_index, CODE_DIR, use_settings_path)
    tau_R_shuffling_CA1 += [tau_R_shuffling/tau_R_bbc]
    tau_R_fivebins_CA1 += [tau_R_fivebins/tau_R_bbc]
    tau_R_onebin_CA1 += [tau_R_onebin/tau_R_bbc]

tau_R_shuffling_CA1_median = np.median(tau_R_shuffling_CA1)
tau_R_shuffling_CA1_median_loCI, tau_R_shuffling_CA1_median_hiCI = plots.get_CI_median(tau_R_shuffling_CA1)
tau_R_fivebins_CA1_median = np.median(tau_R_fivebins_CA1)
tau_R_fivebins_CA1_median_loCI, tau_R_fivebins_CA1_median_hiCI = plots.get_CI_median(tau_R_fivebins_CA1)
tau_R_onebin_CA1_median = np.median(tau_R_onebin_CA1)
tau_R_onebin_CA1_median_loCI, tau_R_onebin_CA1_median_hiCI = plots.get_CI_median(tau_R_onebin_CA1)

recorded_system = 'retina'
number_valid_neurons = 111

tau_R_shuffling_retina = []
tau_R_fivebins_retina = []
tau_R_onebin_retina = []
for neuron_index in range(number_valid_neurons):
    tau_R_bbc, tau_R_shuffling, tau_R_fivebins, tau_R_onebin = get_tau_R_for_different_estimators(T_0, recorded_system, rec_length, neuron_index, CODE_DIR, use_settings_path)
    tau_R_shuffling_retina += [tau_R_shuffling/tau_R_bbc]
    tau_R_fivebins_retina += [tau_R_fivebins/tau_R_bbc]
    tau_R_onebin_retina += [tau_R_onebin/tau_R_bbc]

tau_R_shuffling_retina_median = np.median(tau_R_shuffling_retina)
tau_R_shuffling_retina_median_loCI, tau_R_shuffling_retina_median_hiCI = plots.get_CI_median(tau_R_shuffling_retina)
tau_R_fivebins_retina_median = np.median(tau_R_fivebins_retina)
tau_R_fivebins_retina_median_loCI, tau_R_fivebins_retina_median_hiCI = plots.get_CI_median(tau_R_fivebins_retina)
tau_R_onebin_retina_median = np.median(tau_R_onebin_retina)
tau_R_onebin_retina_median_loCI, tau_R_onebin_retina_median_hiCI = plots.get_CI_median(tau_R_onebin_retina)

recorded_system = 'culture'
number_valid_neurons = 48

tau_R_shuffling_culture = []
tau_R_fivebins_culture = []
tau_R_onebin_culture = []
for neuron_index in range(number_valid_neurons):
    tau_R_bbc, tau_R_shuffling, tau_R_fivebins, tau_R_onebin = get_tau_R_for_different_estimators(T_0, recorded_system, rec_length, neuron_index, CODE_DIR, use_settings_path)
    tau_R_shuffling_culture += [tau_R_shuffling/tau_R_bbc]
    tau_R_fivebins_culture += [tau_R_fivebins/tau_R_bbc]
    tau_R_onebin_culture += [tau_R_onebin/tau_R_bbc]

tau_R_shuffling_culture_median = np.median(tau_R_shuffling_culture)
tau_R_shuffling_culture_median_loCI, tau_R_shuffling_culture_median_hiCI = plots.get_CI_median(tau_R_shuffling_culture)
tau_R_fivebins_culture_median = np.median(tau_R_fivebins_culture)
tau_R_fivebins_culture_median_loCI, tau_R_fivebins_culture_median_hiCI = plots.get_CI_median(tau_R_fivebins_culture)
tau_R_onebin_culture_median = np.median(tau_R_onebin_culture)
tau_R_onebin_culture_median_loCI, tau_R_onebin_culture_median_hiCI = plots.get_CI_median(tau_R_onebin_culture)

# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]
soft_red = sns.color_palette("RdBu_r", 15)[10]
soft_blue = sns.color_palette("RdBu_r", 15)[4]
violet = sns.cubehelix_palette(8)[4]
green = sns.cubehelix_palette(8, start=.5, rot=-.75)[3]

rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '15.0'
matplotlib.rcParams['xtick.labelsize'] = '15'
matplotlib.rcParams['ytick.labelsize'] = '15'
matplotlib.rcParams['legend.fontsize'] = '15'
matplotlib.rcParams['axes.linewidth'] = 0.6

fig, ((ax)) = plt.subplots(1, 1, figsize=(5., 3.))

# ax = plt.subplot2grid((17, 1), (0, 0), colspan=1, rowspan=16)
# ax2 = plt.subplot2grid((17, 1), (17, 0), colspan=1, rowspan=0, sharex=ax)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_ylabel(r'\begin{center} information timescale $\tau_R$ \\ relative to BBC estimate \end{center}')
ax.set_ylim((0.0, 1.2))
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.spines['left'].set_bounds(0.0, 1.)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False)  # labels along the bottom edge are off

# ax2.set_ylim((0.0, 0.05))
# ax2.set_yticks([0.0])
# ax2.spines['left'].set_bounds(0, 0.05)
# ax2.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)

ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xticklabels(
    ['CA1', 'retina', 'culture'], rotation='horizontal')
# ax2.set_xticks(np.array([1, 10, 50]))
x = [1., 4., 7.]
ax.set_xlim((-.5, 8.5))
ax.spines['bottom'].set_bounds(-.5, 8.5)
ax.set_xticks(x)

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
# kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
# ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
# ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

# kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
# ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
# ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

fig.add_subplot(ax)
# fig.add_subplot(ax2)
fig.subplots_adjust(hspace=0.1)

# CA1
ax.bar(x=[0.0], height=[1], yerr=[[0], [0]], width=.5, alpha=.95,
       color=main_red, ecolor="0.3")
ax.bar(x=[0.5], height=[tau_R_shuffling_CA1_median], yerr=[[tau_R_shuffling_CA1_median-tau_R_shuffling_CA1_median_loCI], [tau_R_shuffling_CA1_median_hiCI-tau_R_shuffling_CA1_median]], width=.5, alpha=.95, color=main_blue, ecolor="0.3")
ax.bar(x=[1.0], height=[tau_R_fivebins_CA1_median], yerr=[[tau_R_fivebins_CA1_median-tau_R_fivebins_CA1_median_loCI], [tau_R_fivebins_CA1_median_hiCI-tau_R_fivebins_CA1_median]], width=.5, alpha=.95, color=green, ecolor="0.3")
ax.bar(x=[1.5], height=[tau_R_onebin_CA1_median], yerr=[[tau_R_onebin_CA1_median-tau_R_onebin_CA1_median_loCI], [tau_R_onebin_CA1_median_hiCI-tau_R_onebin_CA1_median]], width=.5, alpha=.95,color='y', ecolor="0.3")

# retina
ax.bar(x=[3.0], height=[1], yerr=[[0], [0]], width=.5, alpha=.95,
       color=main_red, ecolor="0.3")
ax.bar(x=[3.5], height=[tau_R_shuffling_retina_median], yerr=[[tau_R_shuffling_retina_median-tau_R_shuffling_retina_median_loCI], [tau_R_shuffling_retina_median_hiCI - tau_R_shuffling_retina_median]], width=.5, alpha=.95, color=main_blue, ecolor="0.3")
ax.bar(x=[4.0], height=[tau_R_fivebins_retina_median], yerr=[[tau_R_fivebins_retina_median-tau_R_fivebins_retina_median_loCI], [tau_R_fivebins_retina_median_hiCI -tau_R_fivebins_retina_median]], width=.5, alpha=.95, color=green, ecolor="0.3")
ax.bar(x=[4.5], height=[tau_R_onebin_retina_median], yerr=[[tau_R_onebin_retina_median-tau_R_onebin_retina_median_loCI], [tau_R_onebin_retina_median_hiCI-tau_R_onebin_retina_median]], width=.5, alpha=.95,color='y', ecolor="0.3")

# culture
ax.bar(x=[6.0], height=[1], yerr=[[0], [0]], width=.5, alpha=.95,
       color=main_red, ecolor="0.3", label='BBC')
ax.bar(x=[6.5], height=[tau_R_shuffling_culture_median], yerr=[[tau_R_shuffling_culture_median-tau_R_shuffling_culture_median_loCI], [tau_R_shuffling_culture_median_hiCI-tau_R_shuffling_culture_median]], width=.5, alpha=.95,
       color=main_blue, ecolor="0.3", label='Shuffling')
ax.bar(x=[7.0], height=[tau_R_fivebins_culture_median], yerr=[[tau_R_fivebins_culture_median-tau_R_fivebins_culture_median_loCI], [tau_R_fivebins_culture_median_hiCI-tau_R_fivebins_culture_median]], width=.5, alpha=.95, color=green, ecolor="0.3", label='max five bins')
ax.bar(x=[7.5], height=[tau_R_onebin_culture_median], yerr=[[tau_R_onebin_culture_median-tau_R_onebin_culture_median_loCI], [tau_R_onebin_culture_median_hiCI-tau_R_onebin_culture_median]], width=.5, alpha=.95,color='y', ecolor="0.3", label="single bin")

number_valid_neurons = 28
ax.scatter(np.zeros(number_valid_neurons) + 0.6, tau_R_shuffling_CA1,
           s=3, color="0.7", marker="o", zorder=9)
ax.scatter(np.zeros(number_valid_neurons) + 1.1, tau_R_fivebins_CA1,
           s=3, color="0.7", marker="o", zorder=2)
ax.scatter(np.zeros(number_valid_neurons) + 1.6, tau_R_onebin_CA1,
           s=3, color="0.7", marker="o", zorder=2)

number_valid_neurons = 111
ax.scatter(np.zeros(number_valid_neurons) + 3.6, tau_R_shuffling_retina,
           s=3, color="0.7", marker="o", zorder=9)
ax.scatter(np.zeros(number_valid_neurons) + 4.1, tau_R_fivebins_retina,
           s=3, color="0.7", marker="o", zorder=2)
ax.scatter(np.zeros(number_valid_neurons) + 4.6, tau_R_onebin_retina,
           s=3, color="0.7", marker="o", zorder=2)

number_valid_neurons = 48
ax.scatter(np.zeros(number_valid_neurons) + 6.6, tau_R_shuffling_culture,
           s=3, color="0.7", marker="o", zorder=9)
ax.scatter(np.zeros(number_valid_neurons) + 7.1, tau_R_fivebins_culture,
           s=3, color="0.7", marker="o", zorder=2)
ax.scatter(np.zeros(number_valid_neurons) + 7.6, tau_R_onebin_culture,
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

# # CA1
# ax2.bar(x=[0.0], height=[1], yerr=[[0], [0]], width=.5, alpha=.95,
# #        color=main_red, ecolor="0.3")
# ax2.bar(x=[0.5], height=[tau_R_shuffling_CA1_median], yerr=[[tau_R_shuffling_CA1_median-tau_R_shuffling_CA1_median_loCI], [tau_R_shuffling_CA1_median_hiCI-tau_R_shuffling_CA1_median]], width=.5, alpha=.95, color=main_blue, ecolor="0.3")
# ax2.bar(x=[1.0], height=[tau_R_fivebins_CA1_median], yerr=[[tau_R_fivebins_CA1_median-tau_R_fivebins_CA1_median_loCI], [tau_R_fivebins_CA1_median_hiCI-tau_R_fivebins_CA1_median]], width=.5, alpha=.95, color=green, ecolor="0.3")
# ax2.bar(x=[1.5], height=[tau_R_onebin_CA1_median], yerr=[[tau_R_onebin_CA1_median-tau_R_onebin_CA1_median_loCI], [tau_R_onebin_CA1_median_hiCI-tau_R_onebin_CA1_median]], width=.5, alpha=.95,color='y', ecolor="0.3")
# #
# # # retina
# ax2.bar(x=[3.0], height=[1], yerr=[[0], [0]], width=.5, alpha=.95,
# #        color=main_red, ecolor="0.3")
# ax2.bar(x=[3.5], height=[tau_R_shuffling_retina_median], yerr=[[tau_R_shuffling_retina_median-tau_R_shuffling_retina_median_loCI], [tau_R_shuffling_retina_median_hiCI - tau_R_shuffling_retina_median]], width=.5, alpha=.95, color=main_blue, ecolor="0.3")
# ax2.bar(x=[4.0], height=[tau_R_fivebins_retina_median], yerr=[[tau_R_fivebins_retina_median-tau_R_fivebins_retina_median_loCI], [tau_R_fivebins_retina_median_hiCI -tau_R_fivebins_retina_median]], width=.5, alpha=.95, color=green, ecolor="0.3")
# ax2.bar(x=[4.5], height=[tau_R_onebin_retina_median], yerr=[[tau_R_onebin_retina_median-tau_R_onebin_retina_median_loCI], [tau_R_onebin_retina_median_hiCI-tau_R_onebin_retina_median]], width=.5, alpha=.95,color='y', ecolor="0.3")
# #
# # # culture
# ax2.bar(x=[6.0], height=[1], yerr=[[0], [0]], width=.5, alpha=.95,
# #        color=main_red, ecolor="0.3", label='BBC')
# ax2.bar(x=[6.5], height=[tau_R_shuffling_culture_median], yerr=[[tau_R_shuffling_culture_median-tau_R_shuffling_culture_median_loCI], [tau_R_shuffling_culture_median_hiCI-tau_R_shuffling_culture_median]], width=.5, alpha=.95,
# #        color=main_blue, ecolor="0.3", label='Shuffling')
# ax2.bar(x=[7.0], height=[tau_R_fivebins_culture_median], yerr=[[tau_R_fivebins_culture_median-tau_R_fivebins_culture_median_loCI], [tau_R_fivebins_culture_median_hiCI-tau_R_fivebins_culture_median]], width=.5, alpha=.95, color=green, ecolor="0.3", label='max five bins')
# ax2.bar(x=[7.5], height=[tau_R_onebin_culture_median], yerr=[[tau_R_onebin_culture_median-tau_R_onebin_culture_median_loCI], [tau_R_onebin_culture_median_hiCI-tau_R_onebin_culture_median]], width=.5, alpha=.95,color='y', ecolor="0.3", label="single bin")

plt.savefig('%s/S12Fig_tau_R_relative_to_BBC_T0_%s.pdf'%(PLOTTING_DIR, T_0_ms),
            format="pdf", bbox_inches='tight')
plt.show()
plt.close()
