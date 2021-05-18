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

def get_spike_entropy(p_spike):
    p_nospike = 1 - p_spike
    if p_nospike == 1.0:
        entropy = 0
    else:
        entropy = - p_spike * np.log2(p_spike) - p_nospike * np.log2(p_nospike)
    return entropy

"""Global parameters"""
setup = 'fivebins'
regularization_method = 'shuffling'
rec_length = '40min'
total_mutual_information = False
bin_size_ms = 5 #ms, time bin for auto_correlation
# short timescale which is neglected when computing the integrated timescale
# Slightly below 0.01, because the embeddings chosen for analysis are computed for 0.00998, thus almost 0.01 but sligthtly below.
T_0_ms = 10
T_0 = 0.00997

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
    ax.set_xlim((0.3, 12))
    ax.set_xscale('log')
    ax.set_xlabel(r'firing rate (Hz)')

if total_mutual_information == True:
    ax1.set_ylabel(
        r'total mutual information  $I_{\mathrm{tot}}$')
    ax1.set_yscale('log')
    ax1.set_ylim((0.001, 0.1))
    # ax1.set_yticks([0.0, 0.2, 0.4])
    # ax1.spines['left'].set_bounds(.0, .4)
else:
    ax1.set_ylabel(
        r'total history dependence $R_{\mathrm{tot}}$')
    ax1.set_ylim((0.0, 0.45))
    ax1.set_yticks([0.0, 0.2, 0.4])
    ax1.spines['left'].set_bounds(.0, .4)

ax2.set_yscale('log')
ax2.set_ylim((5, 300))
# ax.set_xticks(np.array([1, 10, 50]))
ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.set_ylabel(r'information timescale $\tau_R$ (ms)')


"""Load data"""
recorded_system = 'CA1'
number_valid_neurons = 28
R_tot_CA1 = []
I_tot_CA1 = []
tau_R_CA1 = []
rate_CA1 = []
CV_CA1 = []
medianISI_CA1 = []
autocorrelation_time_CA1 = []
for neuron_index in range(number_valid_neurons):
    tau_R, R_tot = get_tau_R_and_R_tot(T_0, setup, regularization_method, recorded_system, rec_length, neuron_index, CODE_DIR, use_settings_path)
    I_tot = plots.load_total_mutual_information(recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = regularization_method, use_settings_path = use_settings_path)
    stats = get_stats(bin_size_ms, T_0_ms, neuron_index, recorded_system, CODE_DIR)
    rate = stats['rate']
    p_spike = rate*0.005
    H_spiking = get_spike_entropy(p_spike)
    I_tot_rate = R_tot*H_spiking
    R_tot_CA1 += [R_tot]
    I_tot_CA1 += [I_tot_rate]
    tau_R_CA1 += [tau_R*1000]
    rate_CA1 += [stats['rate']]
    medianISI_CA1 += [stats['medianISI']]
    CV_CA1 += [stats['CV']]
    autocorrelation_time_CA1 += [stats['autocorrelation_time']]


R_tot_CA1_median = np.median(R_tot_CA1)
R_tot_CA1_median_loCI, R_tot_CA1_median_hiCI = plots.get_CI_median(R_tot_CA1)
I_tot_CA1_median = np.median(I_tot_CA1)
I_tot_CA1_median_loCI, I_tot_CA1_median_hiCI = plots.get_CI_median(I_tot_CA1)
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
I_tot_retina = []
tau_R_retina = []
rate_retina = []
CV_retina = []
medianISI_retina = []
autocorrelation_time_retina = []
for neuron_index in range(number_valid_neurons):
    tau_R, R_tot = get_tau_R_and_R_tot(T_0, setup, regularization_method, recorded_system, rec_length, neuron_index, CODE_DIR, use_settings_path)
    I_tot = plots.load_total_mutual_information(recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = regularization_method, use_settings_path = use_settings_path)
    stats = get_stats(bin_size_ms, T_0_ms, neuron_index, recorded_system, CODE_DIR)
    rate = stats['rate']
    p_spike = rate*0.005
    H_spiking = get_spike_entropy(p_spike)
    I_tot_rate = R_tot*H_spiking
    R_tot_retina += [R_tot]
    I_tot_retina += [I_tot_rate]
    tau_R_retina += [tau_R*1000]
    rate_retina += [stats['rate']]
    medianISI_retina += [stats['medianISI']]
    CV_retina += [stats['CV']]
    autocorrelation_time_retina += [stats['autocorrelation_time']]

R_tot_retina_median = np.median(R_tot_retina)
R_tot_retina_median_loCI, R_tot_retina_median_hiCI = plots.get_CI_median(R_tot_retina)
I_tot_retina_median = np.median(I_tot_retina)
I_tot_retina_median_loCI, I_tot_retina_median_hiCI = plots.get_CI_median(I_tot_retina)
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
I_tot_culture = []
tau_R_culture = []
rate_culture = []
CV_culture = []
medianISI_culture = []
autocorrelation_time_culture = []
for neuron_index in range(number_valid_neurons):
    tau_R, R_tot = get_tau_R_and_R_tot(T_0, setup, regularization_method, recorded_system, rec_length, neuron_index, CODE_DIR, use_settings_path)
    I_tot = plots.load_total_mutual_information(recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = regularization_method, use_settings_path = use_settings_path)
    stats = get_stats(bin_size_ms, T_0_ms, neuron_index, recorded_system, CODE_DIR)
    rate = stats['rate']
    p_spike = rate*0.005
    H_spiking = get_spike_entropy(p_spike)
    I_tot_rate = R_tot*H_spiking
    R_tot_culture += [R_tot]
    I_tot_culture += [I_tot_rate]
    tau_R_culture += [tau_R*1000]
    rate_culture += [stats['rate']]
    medianISI_culture += [stats['medianISI']]
    CV_culture += [stats['CV']]
    autocorrelation_time_culture += [stats['autocorrelation_time']]

R_tot_culture_median = np.median(R_tot_culture)
R_tot_culture_median_loCI, R_tot_culture_median_hiCI = plots.get_CI_median(R_tot_culture)
I_tot_culture_median = np.median(I_tot_culture)
I_tot_culture_median_loCI, I_tot_culture_median_hiCI = plots.get_CI_median(I_tot_culture)
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
I_tot_V1 = []
tau_R_V1 = []
rate_V1 = []
CV_V1 = []
medianISI_V1 = []
autocorrelation_time_V1 = []
for neuron_index in range(number_valid_neurons):
    tau_R, R_tot = get_tau_R_and_R_tot(T_0, setup, regularization_method, recorded_system, rec_length, neuron_index, CODE_DIR, use_settings_path)
    I_tot = plots.load_total_mutual_information(recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = regularization_method, use_settings_path = use_settings_path)
    stats = get_stats(bin_size_ms, T_0_ms, neuron_index, recorded_system, CODE_DIR)
    rate = stats['rate']
    p_spike = rate*0.005
    H_spiking = get_spike_entropy(p_spike)
    I_tot_rate = R_tot*H_spiking
    R_tot_V1 += [R_tot]
    I_tot_V1 += [I_tot_rate]
    tau_R_V1 += [tau_R*1000]
    rate_V1 += [stats['rate']]
    medianISI_V1 += [stats['medianISI']]
    CV_V1 += [stats['CV']]
    autocorrelation_time_V1 += [stats['autocorrelation_time']]

R_tot_V1_median = np.median(R_tot_V1)
R_tot_V1_median_loCI, R_tot_V1_median_hiCI = plots.get_CI_median(R_tot_V1)
I_tot_V1_median = np.median(I_tot_V1)
I_tot_V1_median_loCI, I_tot_V1_median_hiCI = plots.get_CI_median(I_tot_V1)
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


# I_tot
if total_mutual_information == True:
    ax1.errorbar(x=[rate_culture_median], y=[I_tot_culture_median],  yerr=[[I_tot_culture_median-I_tot_culture_median_loCI], [I_tot_culture_median_hiCI-I_tot_culture_median]], xerr=[[rate_culture_median-rate_culture_median_loCI], [rate_culture_median_hiCI-rate_culture_median]], color=main_red, marker='v', markersize=6)

    ax1.errorbar(x=[rate_retina_median], y=[I_tot_retina_median],  yerr=[[I_tot_retina_median-I_tot_retina_median_loCI], [I_tot_retina_median_hiCI-I_tot_retina_median]], xerr=[[rate_retina_median-rate_retina_median_loCI], [rate_retina_median_hiCI-rate_retina_median]], color='orange', marker='o', markersize=6)

    ax1.errorbar(x=[rate_V1_median], y=[I_tot_V1_median],  yerr=[[I_tot_V1_median-I_tot_V1_median_loCI], [I_tot_V1_median_hiCI-I_tot_V1_median]], xerr=[[rate_V1_median-rate_V1_median_loCI], [rate_V1_median_hiCI-rate_V1_median]], color=green, marker='s', markersize=6)
    ax1.errorbar(x=[rate_CA1_median], y=[I_tot_CA1_median],  yerr=[[I_tot_CA1_median-I_tot_CA1_median_loCI], [I_tot_CA1_median_hiCI-I_tot_CA1_median]], xerr=[[rate_CA1_median-rate_CA1_median_loCI], [rate_CA1_median_hiCI-rate_CA1_median]], color=main_blue, marker='D', markersize=6)
else:
    R_tot
    ax1.errorbar(x=[rate_culture_median], y=[R_tot_culture_median],  yerr=[[R_tot_culture_median-R_tot_culture_median_loCI], [R_tot_culture_median_hiCI-R_tot_culture_median]], xerr=[[rate_culture_median-rate_culture_median_loCI], [rate_culture_median_hiCI-rate_culture_median]], color=main_red, marker='v', markersize=6)

    ax1.errorbar(x=[rate_retina_median], y=[R_tot_retina_median],  yerr=[[R_tot_retina_median-R_tot_retina_median_loCI], [R_tot_retina_median_hiCI-R_tot_retina_median]], xerr=[[rate_retina_median-rate_retina_median_loCI], [rate_retina_median_hiCI-rate_retina_median]], color='orange', marker='o', markersize=6)

    ax1.errorbar(x=[rate_V1_median], y=[R_tot_V1_median],  yerr=[[R_tot_V1_median-R_tot_V1_median_loCI], [R_tot_V1_median_hiCI-R_tot_V1_median]], xerr=[[rate_V1_median-rate_V1_median_loCI], [rate_V1_median_hiCI-rate_V1_median]], color=green, marker='s', markersize=6)

    ax1.errorbar(x=[rate_CA1_median], y=[R_tot_CA1_median],  yerr=[[R_tot_CA1_median-R_tot_CA1_median_loCI], [R_tot_CA1_median_hiCI-R_tot_CA1_median]], xerr=[[rate_CA1_median-rate_CA1_median_loCI], [rate_CA1_median_hiCI-rate_CA1_median]], color=main_blue, marker='D', markersize=6)

# tau_R
ax2.errorbar(x=[rate_culture_median], y=[tau_R_culture_median],  yerr=[[tau_R_culture_median-tau_R_culture_median_loCI], [tau_R_culture_median_hiCI-tau_R_culture_median]], xerr=[[rate_culture_median-rate_culture_median_loCI], [rate_culture_median_hiCI-rate_culture_median]], color=main_red, marker='v', markersize=6)

ax2.errorbar(x=[rate_retina_median], y=[tau_R_retina_median],  yerr=[[tau_R_retina_median-tau_R_retina_median_loCI], [tau_R_retina_median_hiCI-tau_R_retina_median]], xerr=[[rate_retina_median-rate_retina_median_loCI], [rate_retina_median_hiCI-rate_retina_median]], color='orange', marker='o', markersize=6)

ax2.errorbar(x=[rate_V1_median], y=[tau_R_V1_median],  yerr=[[tau_R_V1_median-tau_R_V1_median_loCI], [tau_R_V1_median_hiCI-tau_R_V1_median]], xerr=[[rate_V1_median-rate_V1_median_loCI], [rate_V1_median_hiCI-rate_V1_median]], color=green, marker='s', markersize=6)

ax2.errorbar(x=[rate_CA1_median], y=[tau_R_CA1_median],  yerr=[[tau_R_CA1_median-tau_R_CA1_median_loCI], [tau_R_CA1_median_hiCI-tau_R_CA1_median]], xerr=[[rate_CA1_median-rate_CA1_median_loCI], [rate_CA1_median_hiCI-rate_CA1_median]], color=main_blue, marker='D', markersize=6)

if total_mutual_information == True:
    # Itot
    ax1.scatter(rate_culture, I_tot_culture,
               s=3, color=main_red, marker="v", alpha=0.5, zorder=2)
    ax1.scatter(rate_retina, I_tot_retina,
               s=3, color='orange', marker="o", alpha=0.5, zorder=2)
    ax1.scatter(rate_V1, I_tot_V1,
               s=3, color=green, marker="s", alpha=0.5, zorder=2)
    ax1.scatter(rate_CA1, I_tot_CA1,
               s=3, color=main_blue, marker="s", alpha=0.5, zorder=2)
else:
    # Rtot
    ax1.scatter(rate_culture, R_tot_culture,
               s=3, color=main_red, marker="v", alpha=0.5, zorder=2)
    ax1.scatter(rate_retina, R_tot_retina,
               s=3, color='orange', marker="o", alpha=0.5, zorder=2)
    ax1.scatter(rate_V1, R_tot_V1,
               s=3, color=green, marker="s", alpha=0.5, zorder=2)
    ax1.scatter(rate_CA1, R_tot_CA1,
               s=3, color=main_blue, marker="s", alpha=0.5, zorder=2)

# tau_R
ax2.scatter(rate_culture, tau_R_culture,
           s=3, color=main_red, marker="v", alpha=0.5, zorder=2)
ax2.scatter(rate_retina, tau_R_retina,
           s=3, color='orange', marker="o", alpha=0.5, zorder=2)
ax2.scatter(rate_V1, tau_R_V1,
           s=3, color=green, marker="s", alpha=0.5, zorder=2)
ax2.scatter(rate_CA1, tau_R_CA1,
           s=3, color=main_blue, marker="s", alpha=0.5, zorder=2)


# ax.legend(loc=(1.0, 0.1), frameon=False)
fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

if total_mutual_information == True:
    plt.savefig('%s/S13Fig_Itot_vs_rate.pdf'%(PLOTTING_DIR),
    format="pdf", bbox_inches='tight')
else:
    plt.savefig('%s/S13Fig_measures_vs_rate.pdf'%(PLOTTING_DIR),
                format="pdf", bbox_inches='tight')

plt.show()
plt.close()
