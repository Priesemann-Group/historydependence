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

def get_stats(bin_size_ms, T_0_ms, neuron_index, recorded_system, CODE_DIR):
    ANALYSIS_DIR = '%s/analysis/%s/stats_tbin_%dms'%(CODE_DIR, recorded_system, bin_size_ms)
    with open('%s/stats_neuron%d_T_0_%dms.pkl'%(ANALYSIS_DIR, neuron_index, T_0_ms), 'rb') as f:
        return pickle.load(f)

"""Global parameters"""
setup = 'fivebins'
regularization_method = 'shuffling'
rec_length = '40min'
# short timescale which is neglected when computing the integrated timescale
T_0_ms = int(argv[1]) # 0,10,20
if T_0_ms == 10:
    # Slightly below 0.01, because the embeddings chosen for analysis are computed for 0.00998, thus almost 0.01 but sligthtly below.
    T_0 = 0.00997
else:
    T_0 = T_0_ms/1000.


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

# PLOTTING
# Font settings
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

fig, ((ax)) = plt.subplots(1, 1, figsize=(6.7, 3.2))

##### Unset Borders #####
ax.spines['top'].set_bounds(0, 0)
ax.spines['right'].set_bounds(0, 0)

ax.set_xscale('log')
ax.set_xlim((10, 300))
# ax.set_xticks(np.array([1, 10, 50]))
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.spines['bottom'].set_bounds(10, 300)
ax.set_xlabel(r'information timescale $\tau_{R}$ (ms)')

ax.set_ylabel(
    r'total history dependence $R_{\mathrm{tot}}$')
ax.set_ylim((0.0, 0.45))
ax.set_yticks([0.0, 0.2, 0.4])
ax.spines['left'].set_bounds(.0, .4)

ax.errorbar(x=[tau_R_culture_median], y=[R_tot_culture_median],  yerr=[[R_tot_culture_median-R_tot_culture_median_loCI], [R_tot_culture_median_hiCI-R_tot_culture_median]], xerr=[[tau_R_culture_median-tau_R_culture_median_loCI], [tau_R_culture_median_hiCI-tau_R_culture_median]], color=main_red, marker='v', markersize=6)

ax.errorbar(x=[tau_R_retina_median], y=[R_tot_retina_median],  yerr=[[R_tot_retina_median-R_tot_retina_median_loCI], [R_tot_retina_median_hiCI-R_tot_retina_median]], xerr=[[tau_R_retina_median-tau_R_retina_median_loCI], [tau_R_retina_median_hiCI-tau_R_retina_median]], color='orange', marker='o', markersize=6)

ax.errorbar(x=[tau_R_V1_median], y=[R_tot_V1_median],  yerr=[[R_tot_V1_median-R_tot_V1_median_loCI], [R_tot_V1_median_hiCI-R_tot_V1_median]], xerr=[[tau_R_V1_median-tau_R_V1_median_loCI], [tau_R_V1_median_hiCI-tau_R_V1_median]], color=green, marker='s', markersize=6)

ax.errorbar(x=[tau_R_CA1_median], y=[R_tot_CA1_median],  yerr=[[R_tot_CA1_median-R_tot_CA1_median_loCI], [R_tot_CA1_median_hiCI-R_tot_CA1_median]], xerr=[[tau_R_CA1_median-tau_R_CA1_median_loCI], [tau_R_CA1_median_hiCI-tau_R_CA1_median]], color=main_blue, marker='D', markersize=6)

ax.scatter(x=[tau_R_culture_median], y=[R_tot_culture_median],
           color=main_red, marker='v', s=30, label='rat cortical culture')
ax.scatter(x=[tau_R_retina_median], y=[R_tot_retina_median],
           color='orange', marker='o', s=30, label='salamander retina')
ax.scatter(x=[tau_R_V1_median], y=[R_tot_V1_median],
           color=green, marker='s', s=30, label=r'\begin{center}mouse primary \\ visual cortex\end{center}')
ax.scatter(x=[tau_R_CA1_median], y=[R_tot_CA1_median],
           color=main_blue, marker='D', s=30, label=r'\begin{center}rat dorsal \\ hippocampus (CA1)\end{center}')

ax.scatter(tau_R_culture, R_tot_culture,
           s=3, color=main_red, marker="v", alpha=0.5, zorder=2)
ax.scatter(tau_R_retina, R_tot_retina,
           s=3, color='orange', marker="o", alpha=0.5, zorder=2)
ax.scatter(tau_R_V1, R_tot_V1,
           s=3, color=green, marker="s", alpha=0.5, zorder=2)
ax.scatter(tau_R_CA1, R_tot_CA1,
           s=3, color=main_blue, marker="s", alpha=0.5, zorder=2)

ax.legend(loc=(1.0, 0.1), frameon=False)
fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

plt.savefig('{}/S15Fig_Rtot_vs_tau_R_T0_{}ms.pdf'.format(PLOTTING_DIR,T_0_ms),
            format="pdf", bbox_inches='tight')

plt.show()
plt.close()
