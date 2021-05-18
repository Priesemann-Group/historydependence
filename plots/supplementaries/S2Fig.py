
"""Functions"""
import os
from sys import exit, stderr, argv, path, modules
from os.path import isfile, isdir, realpath, dirname, exists
import numpy as np
import pandas as pd
# plotting
import seaborn.apionly as sns
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

recorded_system = argv[1]
# recorded_system = 'glif_1s_kernel' or 'glif_22s_kernel'
rec_lengths = ['1min', '3min', '5min', '10min', '20min', '45min', '90min']
rec_length_values = [60., 180., 300., 600., 1200., 2700., 5400.]
rec_lengths_colors_bbc = [sns.color_palette("RdBu_r", 15)[8], sns.color_palette("RdBu_r", 15)[9], sns.color_palette("RdBu_r", 15)[10], sns.color_palette(
    "RdBu_r", 15)[11], sns.color_palette("RdBu_r", 15)[12], sns.color_palette("RdBu_r", 15)[13], sns.color_palette("RdBu_r", 15)[14]]
# Only plot first sample

"""Plotting"""
rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '15.0'
matplotlib.rcParams['xtick.labelsize'] = '15'
matplotlib.rcParams['ytick.labelsize'] = '15'
matplotlib.rcParams['legend.fontsize'] = '15'
matplotlib.rcParams['axes.linewidth'] = 0.6

main_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]

"""Load data for all plots"""
mean_R_tot = {}
mean_CI_R_tot = {}

mean_tau_R = {}
mean_CI_tau_R = {}

if recorded_system == 'simulation':
    R_tot_true = np.load('{}/analysis/simulation/R_tot_simulation.npy'.format(CODE_DIR))
else:
    R_tot_true = np.load('{}/analysis/{}/R_tot_900min.npy'.format(CODE_DIR,recorded_system))
T_true, R_true = plots.load_analysis_results_glm_Simulation(CODE_DIR, recorded_system, use_settings_path = use_settings_path)
R_true_running_avg = plots.get_running_avg(R_true)
# dR_true = plots.get_dR(T_true,R_true_running_avg,R_tot_true)[0]
# tau_R_true = plots.get_T_avg(T_true, dR_true, T_0)
dR_true = plots.get_dR(T_true,R_true,R_tot_true)
tau_R_true = plots.get_T_avg(T_true, dR_true, T_0)
print(tau_R_true)

setup = 'full_shuffling'
regularization_method = setup.split("_")[1]
for rec_length in rec_lengths:
    R_tot_arr = []
    tau_R_arr = []
    number_samples = 30
    if rec_length == '45min':
        number_samples = 10
    if rec_length == '90min':
        number_samples = 10
    for sample_index in np.arange(1, number_samples):
        ANALYSIS_DIR, analysis_num_str, R_tot, T_D, T, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(
            recorded_system, rec_length, sample_index, setup, CODE_DIR, regularization_method = regularization_method, use_settings_path=use_settings_path)
        R_tot, tau_R_index, max_valid_index = plots.get_R_tot(T, R, R_CI_lo)
        R_tot_arr += [R_tot]
        # R_running_avg = plots.get_running_avg(R)
        # dR = plots.get_dR(T,R_running_avg,R_tot)[0]
        # tau_R = plots.get_T_avg(T, dR, T_0)
        dR = plots.get_dR(T,R,R_tot)
        tau_R = plots.get_T_avg(T, dR, T_0)
        tau_R_arr += [tau_R]
    mean_R_tot['{}-{}'.format(setup, rec_length)] = np.mean(R_tot_arr)
    mean_CI_R_tot['{}-{}'.format(setup, rec_length)] = plots.get_CI_mean(R_tot_arr)
    mean_tau_R['{}-{}'.format(setup, rec_length)] = np.mean(tau_R_arr)
    mean_CI_tau_R['{}-{}'.format(setup, rec_length)] = plots.get_CI_mean(tau_R_arr)

for rec_length in rec_lengths:
    mean_R_tot['{}-{}'.format(setup, rec_length)] = mean_R_tot['{}-{}'.format(setup, rec_length)]/R_tot_true*100
    mean_CI_R_tot['{}-{}'.format(setup, rec_length)] = mean_CI_R_tot['{}-{}'.format(setup, rec_length)]/R_tot_true*100
    mean_tau_R['{}-{}'.format(setup, rec_length)] = mean_tau_R['{}-{}'.format(setup, rec_length)]/tau_R_true*100
    mean_CI_tau_R['{}-{}'.format(setup, rec_length)] = mean_CI_tau_R['{}-{}'.format(setup, rec_length)]/tau_R_true*100

setup = 'full_bbc'
regularization_method = setup.split("_")[1]
for rec_length in rec_lengths:
    R_tot_arr = []
    tau_R_arr = []
    number_samples = 30
    if rec_length == '45min':
        number_samples = 10
    if rec_length == '90min':
        number_samples = 10
    for sample_index in np.arange(1, number_samples):
        ANALYSIS_DIR, analysis_num_str, R_tot, T_D, T, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(
            recorded_system, rec_length, sample_index, setup, CODE_DIR, regularization_method = regularization_method, use_settings_path=use_settings_path)
        R_tot, tau_R_index, max_valid_index = plots.get_R_tot(T, R, R_CI_lo)
        R_tot_arr += [R_tot]
        # R_running_avg = plots.get_running_avg(R)
        # dR = plots.get_dR(T,R_running_avg,R_tot)
        # tau_R = plots.get_T_avg(T, dR, T_0)
        dR = plots.get_dR(T,R,R_tot)
        tau_R = plots.get_T_avg(T, dR, T_0)
        tau_R_arr += [tau_R]
        # print(tau_R, tau_R_long)
    mean_R_tot['{}-{}'.format(setup, rec_length)] = np.mean(R_tot_arr)
    mean_CI_R_tot['{}-{}'.format(setup, rec_length)] = plots.get_CI_mean(R_tot_arr)
    mean_tau_R['{}-{}'.format(setup, rec_length)] = np.mean(tau_R_arr)
    mean_CI_tau_R['{}-{}'.format(setup, rec_length)] = plots.get_CI_mean(tau_R_arr)

for rec_length in rec_lengths:
    mean_R_tot['{}-{}'.format(setup, rec_length)] = mean_R_tot['{}-{}'.format(setup, rec_length)]/R_tot_true*100
    mean_CI_R_tot['{}-{}'.format(setup, rec_length)] = mean_CI_R_tot['{}-{}'.format(setup, rec_length)]/R_tot_true*100
    mean_tau_R['{}-{}'.format(setup, rec_length)] = mean_tau_R['{}-{}'.format(setup, rec_length)]/tau_R_true*100
    mean_CI_tau_R['{}-{}'.format(setup, rec_length)] = mean_CI_tau_R['{}-{}'.format(setup, rec_length)]/tau_R_true*100

"""Plotting"""
fig, (axes) = plt.subplots(1, 2, figsize=(10, 3.2))
for j, ax in enumerate(axes):

    x = rec_length_values
    labels = ['1', '3', '5', '10', '20', '45', '90']

    ax.set_xscale('log')
    ax.set_xlim((50, 22500))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.spines['bottom'].set_bounds(50, 5500)
    ax.minorticks_off()
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r'recording time $T_{\mathrm{rec}}$ (min)')

    ##### Unset Borders #####
    ax.spines['top'].set_bounds(0, 0)
    ax.spines['right'].set_bounds(0, 0)
    ax.plot(rec_length_values, np.zeros(len(rec_lengths))+100, linestyle = '--', color='0')
    # only plot labels and legend for left-hand side
    if j == 0:
        ax.set_ylabel(r'\begin{center}total history dependence $\hat{R}_{\mathrm{tot}}$ \\ relative to true $R_{\mathrm{tot}}$ (\%)\end{center}')
    if j == 1:
        ax.set_ylabel(r'\begin{center}information timescale $\hat{\tau}_{R}$\\ relative to true $\tau_R$ (\%)\end{center}')

    setup = 'full_bbc'
    for i, rec_length in enumerate(rec_lengths):
        if j == 0:
            mean_val = mean_R_tot['{}-{}'.format(
            setup, rec_length)]
            mean_CI_val = mean_CI_R_tot['{}-{}'.format(
            setup, rec_length)]
            ax.set_ylim((0, 105.5))
            ax.spines['left'].set_bounds(0, 100)
        if j == 1:
            ax.set_ylim((0, 105.5))
            ax.spines['left'].set_bounds(0, 100)
            mean_val = mean_tau_R['{}-{}'.format(
            setup, rec_length)]
            mean_CI_val = mean_CI_tau_R['{}-{}'.format(
            setup, rec_length)]

        mean_CI_lo = mean_CI_val[0] - mean_val
        mean_CI_hi = mean_CI_val[1] - mean_val
        if i == 0:
            ax.errorbar(x=[rec_length_values[i]], y=[mean_val], yerr=[[-mean_CI_lo], [mean_CI_hi]],
                    color=main_red, marker='d', markersize=5.,capsize = 3.0, label = r'BBC, $d_{\mathrm{max}}=25$')
        else:
            ax.errorbar(x=[rec_length_values[i]], y=[mean_val], yerr=[[-mean_CI_lo], [mean_CI_hi]],
                        color=main_red, marker='d', markersize=5.,capsize = 3.0)

    setup = 'full_shuffling'
    for i, rec_length in enumerate(rec_lengths):
        if j == 0:
            mean_val = mean_R_tot['{}-{}'.format(
            setup, rec_length)]
            mean_CI_val = mean_CI_R_tot['{}-{}'.format(
            setup, rec_length)]
            ax.set_ylim((0, 105.5))
            ax.spines['left'].set_bounds(0, 100)
        if j == 1:
            ax.set_ylim((0, 105.5))
            ax.spines['left'].set_bounds(0, 100)
            mean_val = mean_tau_R['{}-{}'.format(
            setup, rec_length)]
            mean_CI_val = mean_CI_tau_R['{}-{}'.format(
            setup, rec_length)]

        mean_CI_lo = mean_CI_val[0] - mean_val
        mean_CI_hi = mean_CI_val[1] - mean_val
        if i == 0:
            ax.errorbar(x=[rec_length_values[i]], y=[mean_val], yerr=[[-mean_CI_lo], [mean_CI_hi]],
                            color=main_blue, marker='d', markersize=5., label = r'Shuffling, $d_{\mathrm{max}}=25$')
        else:
            ax.errorbar(x=[rec_length_values[i]], y=[mean_val], yerr=[[-mean_CI_lo], [mean_CI_hi]],
                        color=main_blue, marker='d', markersize=5.)



    if j == 0:
        ax.legend(loc=(0.1, 0.1), frameon=False)

# fig.text(0.5, 1.01, r'simulation', ha='center', va='center', fontsize = 20)
fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

# plt.savefig('{}/S2Fig_tau_R.png'.format(PLOTTING_DIR),
#             format="png",dpi = 600, bbox_inches='tight')
plt.savefig('{}/S2Fig_{}.pdf'.format(PLOTTING_DIR, recorded_system),
            format="pdf", bbox_inches='tight')
plt.show()
plt.close()
