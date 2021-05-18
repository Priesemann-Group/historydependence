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

def get_tau_R_and_R_tot(T_0, setup, regularization_method, recorded_system, rec_length, neuron_index, CODE_DIR, use_settings_path):
    ANALYSIS_DIR, analysis_num_str, R_tot, T_D, T, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = regularization_method, use_settings_path = use_settings_path)
    R_tot = plots.get_R_tot(T, R, R_CI_lo)[0]
    dR = plots.get_dR(T,R,R_tot)
    tau_R = plots.get_T_avg(T, dR, T_0)
    return tau_R, R_tot


"""Global parameters"""
setup = 'fivebins_1ms'
regularization_method = 'shuffling'
rec_length = '40min'
# short timescale which is neglected when computing the integrated timescale
# Slightly below 0.01, because the embeddings chosen for analysis are computed for 0.00998, thus almost 0.01 but sligthtly below.
T_0_ms = 10
T_0 = 0.00997
# T_0 = 0

rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '16.0'
matplotlib.rcParams['xtick.labelsize'] = '16'
matplotlib.rcParams['ytick.labelsize'] = '16'
matplotlib.rcParams['legend.fontsize'] = '16'
matplotlib.rcParams['axes.linewidth'] = 0.6
matplotlib.rcParams["errorbar.capsize"] = 2.5

# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]
soft_red = sns.color_palette("RdBu_r", 15)[10]
soft_blue = sns.color_palette("RdBu_r", 15)[4]
violet = sns.cubehelix_palette(8)[4]
green = sns.cubehelix_palette(8, start=.5, rot=-.75)[3]

fig, ((ax1,ax2)) = plt.subplots(1, 2, figsize=(7, 3.2))

##### Unset Borders #####


for ax in (ax1,ax2):
    ax.spines['top'].set_bounds(0, 0)
    ax.spines['right'].set_bounds(0, 0)
    ax.set_xlim((0,10.5))
    ax.set_xticks([1,5,10])
    ax.set_xlabel(r'time bin $\Delta t$ (ms)')

ax1.set_ylabel(
    r'total history dependence $R_{\mathrm{tot}}$')
ax1.set_ylim((0.0, 0.45))
ax1.set_yticks([0.0, 0.2, 0.4])
ax1.spines['left'].set_bounds(.0, .4)

ax2.set_yscale('log')
ax2.set_ylim((10, 300))
# ax.set_xticks(np.array([1, 10, 50]))
ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.set_ylabel(r'information timescale $\tau_R$ (ms)')

for t_bin in [1,2,3,4,5,7,10]:
    setup = 'fivebins_%dms'%t_bin
    if t_bin == 5:
        setup = 'fivebins'

    """Load data"""
    recorded_system = 'CA1'
    number_valid_neurons = 28
    R_tot_CA1 = []
    tau_R_CA1 = []
    for neuron_index in range(number_valid_neurons):
        tau_R, R_tot = get_tau_R_and_R_tot(T_0, setup, regularization_method, recorded_system, rec_length, neuron_index, CODE_DIR, use_settings_path)
        R_tot_CA1 += [R_tot]
        tau_R_CA1 += [tau_R*1000]

    R_tot_CA1_median = np.median(R_tot_CA1)
    R_tot_CA1_median_loCI, R_tot_CA1_median_hiCI = plots.get_CI_median(R_tot_CA1)
    tau_R_CA1_median = np.median(tau_R_CA1)
    tau_R_CA1_median_loCI, tau_R_CA1_median_hiCI = plots.get_CI_median(tau_R_CA1)

    recorded_system = 'retina'
    number_valid_neurons = 111
    R_tot_retina = []
    tau_R_retina = []
    for neuron_index in range(number_valid_neurons):
        tau_R, R_tot = get_tau_R_and_R_tot(T_0, setup, regularization_method, recorded_system, rec_length, neuron_index, CODE_DIR, use_settings_path)
        R_tot_retina += [R_tot]
        tau_R_retina += [tau_R*1000]

    R_tot_retina_median = np.median(R_tot_retina)
    R_tot_retina_median_loCI, R_tot_retina_median_hiCI = plots.get_CI_median(R_tot_retina)
    tau_R_retina_median = np.median(tau_R_retina)
    tau_R_retina_median_loCI, tau_R_retina_median_hiCI = plots.get_CI_median(tau_R_retina)

    recorded_system = 'culture'
    number_valid_neurons = 48
    R_tot_culture = []
    tau_R_culture = []
    for neuron_index in range(number_valid_neurons):
        tau_R, R_tot = get_tau_R_and_R_tot(T_0, setup, regularization_method, recorded_system, rec_length, neuron_index, CODE_DIR, use_settings_path)
        R_tot_culture += [R_tot]
        tau_R_culture += [tau_R*1000]

    R_tot_culture_median = np.median(R_tot_culture)
    R_tot_culture_median_loCI, R_tot_culture_median_hiCI = plots.get_CI_median(R_tot_culture)
    tau_R_culture_median = np.median(tau_R_culture)
    tau_R_culture_median_loCI, tau_R_culture_median_hiCI = plots.get_CI_median(tau_R_culture)

    recorded_system = 'V1'
    number_valid_neurons = 142
    R_tot_V1 = []
    tau_R_V1 = []
    for neuron_index in range(number_valid_neurons):
        tau_R, R_tot = get_tau_R_and_R_tot(T_0, setup, regularization_method, recorded_system, rec_length, neuron_index, CODE_DIR, use_settings_path)
        R_tot_V1 += [R_tot]
        tau_R_V1 += [tau_R*1000]

    R_tot_V1_median = np.median(R_tot_V1)
    R_tot_V1_median_loCI, R_tot_V1_median_hiCI = plots.get_CI_median(R_tot_V1)
    tau_R_V1_median = np.median(tau_R_V1)
    tau_R_V1_median_loCI, tau_R_V1_median_hiCI = plots.get_CI_median(tau_R_V1)


    ax1.errorbar(x=[t_bin], y=[R_tot_culture_median],  yerr=[[R_tot_culture_median-R_tot_culture_median_loCI], [R_tot_culture_median_hiCI-R_tot_culture_median]], color=main_red, marker='v', markersize=6)

    ax1.errorbar(x=[t_bin], y=[R_tot_retina_median],  yerr=[[R_tot_retina_median-R_tot_retina_median_loCI], [R_tot_retina_median_hiCI-R_tot_retina_median]], color='orange', marker='o', markersize=6)

    ax1.errorbar(x=[t_bin], y=[R_tot_V1_median],  yerr=[[R_tot_V1_median-R_tot_V1_median_loCI], [R_tot_V1_median_hiCI-R_tot_V1_median]], color=green, marker='s', markersize=6)

    ax1.errorbar(x=[t_bin], y=[R_tot_CA1_median],  yerr=[[R_tot_CA1_median-R_tot_CA1_median_loCI], [R_tot_CA1_median_hiCI-R_tot_CA1_median]], color=main_blue, marker='D', markersize=6)

    ax2.errorbar(x=[t_bin], y=[tau_R_culture_median],  yerr=[[tau_R_culture_median-tau_R_culture_median_loCI], [tau_R_culture_median_hiCI-tau_R_culture_median]], color=main_red, marker='v', markersize=6)

    ax2.errorbar(x=[t_bin], y=[tau_R_retina_median],  yerr=[[tau_R_retina_median-tau_R_retina_median_loCI], [tau_R_retina_median_hiCI-tau_R_retina_median]], color='orange', marker='o', markersize=6)

    ax2.errorbar(x=[t_bin], y=[tau_R_V1_median],  yerr=[[tau_R_V1_median-tau_R_V1_median_loCI], [tau_R_V1_median_hiCI-tau_R_V1_median]], color=green, marker='s', markersize=6)

    ax2.errorbar(x=[t_bin], y=[tau_R_CA1_median],  yerr=[[tau_R_CA1_median-tau_R_CA1_median_loCI], [tau_R_CA1_median_hiCI-tau_R_CA1_median]], color=main_blue, marker='D', markersize=6)

    # ax.scatter(x=[tau_R_culture_median], y=[R_tot_culture_median],
    #            color=main_red, marker='v', s=30, label='rat cortical culture')
    # ax.scatter(x=[tau_R_retina_median], y=[R_tot_retina_median],
    #            color='orange', marker='o', s=30, label='salamander retina')
    # ax.scatter(x=[tau_R_V1_median], y=[R_tot_V1_median],
    #            color=green, marker='s', s=30, label=r'\begin{center}mouse primary \\ visual cortex\end{center}')
    # ax.scatter(x=[tau_R_CA1_median], y=[R_tot_CA1_median],
    #            color=main_blue, marker='D', s=30, label=r'\begin{center}rat dorsal \\ hippocampus (CA1)\end{center}')

    # ax1.scatter(np.zeros(len(R_tot_culture))+t_bin, R_tot_culture,
    #            s=3, color=main_red, marker="v", alpha=0.5, zorder=2)
    # ax1.scatter(np.zeros(len(R_tot_retina))+t_bin, R_tot_retina,
    #            s=3, color='orange', marker="o", alpha=0.5, zorder=2)
    # ax1.scatter(np.zeros(len(R_tot_V1))+t_bin, R_tot_V1,
    #            s=3, color=green, marker="s", alpha=0.5, zorder=2)
    # ax1.scatter(np.zeros(len(R_tot_CA1))+t_bin, R_tot_CA1,
    #            s=3, color=main_blue, marker="s", alpha=0.5, zorder=2)
    #
    # ax2.scatter(np.zeros(len(tau_R_culture))+t_bin, tau_R_culture,
    #            s=3, color=main_red, marker="v", alpha=0.5, zorder=2)
    # ax2.scatter(np.zeros(len(tau_R_retina))+t_bin, tau_R_retina,
    #            s=3, color='orange', marker="o", alpha=0.5, zorder=2)
    # ax2.scatter(np.zeros(len(tau_R_V1))+t_bin, tau_R_V1,
    #            s=3, color=green, marker="s", alpha=0.5, zorder=2)
    # ax2.scatter(np.zeros(len(tau_R_CA1))+t_bin, tau_R_CA1,
    #            s=3, color=main_blue, marker="s", alpha=0.5, zorder=2)

# ax.legend(loc=(1.0, 0.1), frameon=False)
fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

plt.savefig('%s/S16Fig_measures_vs_t_bin.pdf'%(PLOTTING_DIR),
            format="pdf", bbox_inches='tight')
# plt.savefig('{}/Fig5_tau_R_T0_10ms.png_1ms'.format(PLOTTING_DIR),
#             format="png", dpi = 600, bbox_inches='tight')

plt.show()
plt.close()
