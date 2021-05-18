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
import pickle
from scipy.stats import pearsonr
from scipy.stats import spearmanr

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
bin_size_ms = 5 #ms, time bin for auto_correlation
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
    ax.set_xlim((10, 2000))
    ax.set_xscale('log')
    ax.set_xticks([10,100,1000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.1,0.2, 0.3, 0.4, 0.5,0.6,0.7, 0.8, 0.9),numticks=12)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r'autocorrelation time $\tau_C$ (ms)')

ax1.set_ylabel(
    r'total history dependence $R_{\mathrm{tot}}$')
ax1.set_ylim((0.0, 0.45))
ax1.set_yticks([0.0, 0.2, 0.4])
ax1.spines['left'].set_bounds(.0, .4)

ax2.set_yscale('log')
ax2.set_ylim((5, 300))
# ax.set_xticks(np.array([1, 10, 50]))
ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.set_ylabel(r'information timescale $\tau_{R}$ (ms)')


"""Load data"""
recorded_system = 'CA1'
number_valid_neurons = 28
R_tot_CA1 = []
tau_R_CA1 = []
rate_CA1 = []
CV_CA1 = []
medianISI_CA1 = []
autocorrelation_time_CA1 = []
for neuron_index in range(number_valid_neurons):
    tau_R, R_tot = get_tau_R_and_R_tot(T_0, setup, regularization_method, recorded_system, rec_length, neuron_index, CODE_DIR, use_settings_path)
    stats = get_stats(bin_size_ms, T_0_ms, neuron_index, recorded_system, CODE_DIR)
    R_tot_CA1 += [R_tot]
    tau_R_CA1 += [tau_R*1000]
    rate_CA1 += [stats['rate']]
    medianISI_CA1 += [stats['medianISI']]
    CV_CA1 += [stats['CV']]
    autocorrelation_time_CA1 += [stats['autocorrelation_time']]

R_tot_CA1_median = np.median(R_tot_CA1)
R_tot_CA1_median_loCI, R_tot_CA1_median_hiCI = plots.get_CI_median(R_tot_CA1)
tau_R_CA1_median = np.median(tau_R_CA1)
tau_R_CA1_median_loCI, tau_R_CA1_median_hiCI = plots.get_CI_median(tau_R_CA1)
rate_CA1_median = np.median(rate_CA1)
rate_CA1_median_loCI, rate_CA1_median_hiCI = plots.get_CI_median(rate_CA1)
medianISI_CA1_median = np.median(medianISI_CA1)
medianISI_CA1_median_loCI, medianISI_CA1_median_hiCI = plots.get_CI_median(medianISI_CA1)
CV_CA1_median = np.median(CV_CA1)
CV_CA1_median_loCI, CV_CA1_median_hiCI = plots.get_CI_median(CV_CA1)
autocorrelation_time_CA1_median = np.median(autocorrelation_time_CA1)
autocorrelation_time_CA1_median_loCI, autocorrelation_time_CA1_median_hiCI = plots.get_CI_median(autocorrelation_time_CA1)

recorded_system = 'retina'
number_valid_neurons = 111
R_tot_retina = []
tau_R_retina = []
rate_retina = []
CV_retina = []
medianISI_retina = []
autocorrelation_time_retina = []
for neuron_index in range(number_valid_neurons):
    tau_R, R_tot = get_tau_R_and_R_tot(T_0, setup, regularization_method, recorded_system, rec_length, neuron_index, CODE_DIR, use_settings_path)
    stats = get_stats(bin_size_ms, T_0_ms, neuron_index, recorded_system, CODE_DIR)
    R_tot_retina += [R_tot]
    tau_R_retina += [tau_R*1000]
    rate_retina += [stats['rate']]
    medianISI_retina += [stats['medianISI']]
    CV_retina += [stats['CV']]
    autocorrelation_time_retina += [stats['autocorrelation_time']]

R_tot_retina_median = np.median(R_tot_retina)
R_tot_retina_median_loCI, R_tot_retina_median_hiCI = plots.get_CI_median(R_tot_retina)
tau_R_retina_median = np.median(tau_R_retina)
tau_R_retina_median_loCI, tau_R_retina_median_hiCI = plots.get_CI_median(tau_R_retina)
rate_retina_median = np.median(rate_retina)
rate_retina_median_loCI, rate_retina_median_hiCI = plots.get_CI_median(rate_retina)
medianISI_retina_median = np.median(medianISI_retina)
medianISI_retina_median_loCI, medianISI_retina_median_hiCI = plots.get_CI_median(medianISI_retina)
CV_retina_median = np.median(CV_retina)
CV_retina_median_loCI, CV_retina_median_hiCI = plots.get_CI_median(CV_retina)
autocorrelation_time_retina_median = np.median(autocorrelation_time_retina)
autocorrelation_time_retina_median_loCI, autocorrelation_time_retina_median_hiCI = plots.get_CI_median(autocorrelation_time_retina)

recorded_system = 'culture'
number_valid_neurons = 48
R_tot_culture = []
tau_R_culture = []
rate_culture = []
CV_culture = []
medianISI_culture = []
autocorrelation_time_culture = []
for neuron_index in range(number_valid_neurons):
    tau_R, R_tot = get_tau_R_and_R_tot(T_0, setup, regularization_method, recorded_system, rec_length, neuron_index, CODE_DIR, use_settings_path)
    stats = get_stats(bin_size_ms, T_0_ms, neuron_index, recorded_system, CODE_DIR)
    R_tot_culture += [R_tot]
    tau_R_culture += [tau_R*1000]
    rate_culture += [stats['rate']]
    medianISI_culture += [stats['medianISI']]
    CV_culture += [stats['CV']]
    autocorrelation_time_culture += [stats['autocorrelation_time']]

R_tot_culture_median = np.median(R_tot_culture)
R_tot_culture_median_loCI, R_tot_culture_median_hiCI = plots.get_CI_median(R_tot_culture)
tau_R_culture_median = np.median(tau_R_culture)
tau_R_culture_median_loCI, tau_R_culture_median_hiCI = plots.get_CI_median(tau_R_culture)
rate_culture_median = np.median(rate_culture)
rate_culture_median_loCI, rate_culture_median_hiCI = plots.get_CI_median(rate_culture)
medianISI_culture_median = np.median(medianISI_culture)
medianISI_culture_median_loCI, medianISI_culture_median_hiCI = plots.get_CI_median(medianISI_culture)
CV_culture_median = np.median(CV_culture)
CV_culture_median_loCI, CV_culture_median_hiCI = plots.get_CI_median(CV_culture)
autocorrelation_time_culture_median = np.median(autocorrelation_time_culture)
autocorrelation_time_culture_median_loCI, autocorrelation_time_culture_median_hiCI = plots.get_CI_median(autocorrelation_time_culture)


recorded_system = 'V1'
number_valid_neurons = 142
R_tot_V1 = []
tau_R_V1 = []
rate_V1 = []
CV_V1 = []
medianISI_V1 = []
autocorrelation_time_V1 = []
for neuron_index in range(number_valid_neurons):
    tau_R, R_tot = get_tau_R_and_R_tot(T_0, setup, regularization_method, recorded_system, rec_length, neuron_index, CODE_DIR, use_settings_path)
    stats = get_stats(bin_size_ms, T_0_ms, neuron_index, recorded_system, CODE_DIR)
    R_tot_V1 += [R_tot]
    tau_R_V1 += [tau_R*1000]
    rate_V1 += [stats['rate']]
    medianISI_V1 += [stats['medianISI']]
    CV_V1 += [stats['CV']]
    autocorrelation_time_V1 += [stats['autocorrelation_time']]

R_tot_V1_median = np.median(R_tot_V1)
R_tot_V1_median_loCI, R_tot_V1_median_hiCI = plots.get_CI_median(R_tot_V1)
tau_R_V1_median = np.median(tau_R_V1)
tau_R_V1_median_loCI, tau_R_V1_median_hiCI = plots.get_CI_median(tau_R_V1)
rate_V1_median = np.median(rate_V1)
rate_V1_median_loCI, rate_V1_median_hiCI = plots.get_CI_median(rate_V1)
medianISI_V1_median = np.median(medianISI_V1)
medianISI_V1_median_loCI, medianISI_V1_median_hiCI = plots.get_CI_median(medianISI_V1)
CV_V1_median = np.median(CV_V1)
CV_V1_median_loCI, CV_V1_median_hiCI = plots.get_CI_median(CV_V1)
autocorrelation_time_V1_median = np.median(autocorrelation_time_V1)
autocorrelation_time_V1_median_loCI, autocorrelation_time_V1_median_hiCI = plots.get_CI_median(autocorrelation_time_V1)

# linear relation
r_retina, p_value_retina = pearsonr(tau_R_retina, autocorrelation_time_retina)
r_CA1, p_value_CA1 = pearsonr(tau_R_CA1, autocorrelation_time_CA1)
r_culture, p_value_culture = pearsonr(tau_R_culture, autocorrelation_time_culture)
r_V1, p_value_V1 = pearsonr(tau_R_V1, autocorrelation_time_V1)
print('Pearson')
print('retina: ', r_retina, p_value_retina)
print('CA1: ', r_CA1, p_value_CA1)
print('culture: ', r_culture, p_value_culture)
print('V1: ', r_V1, p_value_V1)

# rank relation
r_retina, p_value_retina = spearmanr(tau_R_retina, autocorrelation_time_retina)
r_CA1, p_value_CA1 = spearmanr(tau_R_CA1, autocorrelation_time_CA1)
r_culture, p_value_culture = spearmanr(tau_R_culture, autocorrelation_time_culture)
r_V1, p_value_V1 = spearmanr(tau_R_V1, autocorrelation_time_V1)
print('Spearman')
print('retina: ', r_retina, p_value_retina)
print('CA1: ', r_CA1, p_value_CA1)
print('culture: ', r_culture, p_value_culture)
print('V1: ', r_V1, p_value_V1)


ax1.errorbar(x=[autocorrelation_time_culture_median], y=[R_tot_culture_median],  yerr=[[R_tot_culture_median-R_tot_culture_median_loCI], [R_tot_culture_median_hiCI-R_tot_culture_median]], xerr=[[autocorrelation_time_culture_median-autocorrelation_time_culture_median_loCI], [autocorrelation_time_culture_median_hiCI-autocorrelation_time_culture_median]], color=main_red, marker='v', markersize=6)

ax1.errorbar(x=[autocorrelation_time_retina_median], y=[R_tot_retina_median],  yerr=[[R_tot_retina_median-R_tot_retina_median_loCI], [R_tot_retina_median_hiCI-R_tot_retina_median]], xerr=[[autocorrelation_time_retina_median-autocorrelation_time_retina_median_loCI], [autocorrelation_time_retina_median_hiCI-autocorrelation_time_retina_median]], color='orange', marker='o', markersize=6)

ax1.errorbar(x=[autocorrelation_time_V1_median], y=[R_tot_V1_median],  yerr=[[R_tot_V1_median-R_tot_V1_median_loCI], [R_tot_V1_median_hiCI-R_tot_V1_median]], xerr=[[autocorrelation_time_V1_median-autocorrelation_time_V1_median_loCI], [autocorrelation_time_V1_median_hiCI-autocorrelation_time_V1_median]], color=green, marker='s', markersize=6)

ax1.errorbar(x=[autocorrelation_time_CA1_median], y=[R_tot_CA1_median],  yerr=[[R_tot_CA1_median-R_tot_CA1_median_loCI], [R_tot_CA1_median_hiCI-R_tot_CA1_median]], xerr=[[autocorrelation_time_CA1_median-autocorrelation_time_CA1_median_loCI], [autocorrelation_time_CA1_median_hiCI-autocorrelation_time_CA1_median]], color=main_blue, marker='D', markersize=6)

ax2.errorbar(x=[autocorrelation_time_culture_median], y=[tau_R_culture_median],  yerr=[[tau_R_culture_median-tau_R_culture_median_loCI], [tau_R_culture_median_hiCI-tau_R_culture_median]], xerr=[[autocorrelation_time_culture_median-autocorrelation_time_culture_median_loCI], [autocorrelation_time_culture_median_hiCI-autocorrelation_time_culture_median]], color=main_red, marker='v', markersize=6)

ax2.errorbar(x=[autocorrelation_time_retina_median], y=[tau_R_retina_median],  yerr=[[tau_R_retina_median-tau_R_retina_median_loCI], [tau_R_retina_median_hiCI-tau_R_retina_median]], xerr=[[autocorrelation_time_retina_median-autocorrelation_time_retina_median_loCI], [autocorrelation_time_retina_median_hiCI-autocorrelation_time_retina_median]], color='orange', marker='o', markersize=6)

ax2.errorbar(x=[autocorrelation_time_V1_median], y=[tau_R_V1_median],  yerr=[[tau_R_V1_median-tau_R_V1_median_loCI], [tau_R_V1_median_hiCI-tau_R_V1_median]], xerr=[[autocorrelation_time_V1_median-autocorrelation_time_V1_median_loCI], [autocorrelation_time_V1_median_hiCI-autocorrelation_time_V1_median]], color=green, marker='s', markersize=6)

ax2.errorbar(x=[autocorrelation_time_CA1_median], y=[tau_R_CA1_median],  yerr=[[tau_R_CA1_median-tau_R_CA1_median_loCI], [tau_R_CA1_median_hiCI-tau_R_CA1_median]], xerr=[[autocorrelation_time_CA1_median-autocorrelation_time_CA1_median_loCI], [autocorrelation_time_CA1_median_hiCI-autocorrelation_time_CA1_median]], color=main_blue, marker='D', markersize=6)

ax1.scatter(autocorrelation_time_culture, R_tot_culture,
           s=3, color=main_red, marker="v", alpha=0.5, zorder=2)
ax1.scatter(autocorrelation_time_retina, R_tot_retina,
           s=3, color='orange', marker="o", alpha=0.5, zorder=2)
ax1.scatter(autocorrelation_time_V1, R_tot_V1,
           s=3, color=green, marker="s", alpha=0.5, zorder=2)
ax1.scatter(autocorrelation_time_CA1, R_tot_CA1,
           s=3, color=main_blue, marker="s", alpha=0.5, zorder=2)

ax2.scatter(autocorrelation_time_culture, tau_R_culture,
           s=3, color=main_red, marker="v", alpha=0.5, zorder=2)
ax2.scatter(autocorrelation_time_retina, tau_R_retina,
           s=3, color='orange', marker="o", alpha=0.5, zorder=2)
ax2.scatter(autocorrelation_time_V1, tau_R_V1,
           s=3, color=green, marker="s", alpha=0.5, zorder=2)
ax2.scatter(autocorrelation_time_CA1, tau_R_CA1,
           s=3, color=main_blue, marker="s", alpha=0.5, zorder=2)


# ax.legend(loc=(0.1, 1.0), frameon=False)
fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

plt.savefig('%s/S14Fig_measures_vs_autocorrelation_time.pdf'%(PLOTTING_DIR),
            format="pdf", bbox_inches='tight')
plt.show()
plt.close()
