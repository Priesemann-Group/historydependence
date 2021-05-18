
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
setups = np.array([['full_bbc','full_bbc_withCV'], ['full_shuffling','full_shuffling_withCV']])
# Only plot first sample

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

"""Load data for all plots"""
mean_rel_deviation_R_tot = {}
mean_CI_rel_deviation_R_tot = {}

for setup in setups.flatten():
    regularization_method = setup.split("_")[1]
    for rec_length in rec_lengths:
        rel_deviation_R_tot = []
        number_samples = 30
        if rec_length == '45min':
            number_samples = 10
        if rec_length == '90min':
            number_samples = 10
        for sample_index in np.arange(1, number_samples):
            ANALYSIS_DIR, analysis_num_str, R_tot, T_D, T, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(
                recorded_system, rec_length, sample_index, setup, CODE_DIR, regularization_method = regularization_method, use_settings_path = use_settings_path)
            R_tot, T_D_index, max_valid_index = plots.get_R_tot(T, R, R_CI_lo)

            glm_csv_file_name = '{}/ANALYSIS{}/glm_benchmark_{}.csv'.format(
                ANALYSIS_DIR, analysis_num_str, regularization_method)
            glm_pd = pd.read_csv(glm_csv_file_name)
            T_glm = np.array(glm_pd['T'])
            R_glm = np.array(glm_pd['R_GLM'])
            # Make sure that you only average R_GLM over the right T
            if T_glm[0]>T[T_D_index]:
                print(setup, rec_length, sample_index)
            else:
                T_D_index_glm = np.where(T_glm == T[T_D_index])[0][0]
                max_valid_index_glm = np.where(T_glm == T[max_valid_index-1])[0][0]+1
                # Estimate the mean for Rtot, based on the benchmark estimates with the glm
                R_tot_glm = np.mean(R_glm[T_D_index_glm:max_valid_index_glm])
                # The bias is computed as the relative difference between the benchmark value of Rtot, and the estimated value of Rtot
                rel_deviation_R_tot += [100* (R_tot-R_tot_glm)/R_tot_glm]

        mean_rel_deviation_R_tot['{}-{}'.format(setup, rec_length)
                          ] = np.mean(rel_deviation_R_tot)
        mean_CI_rel_deviation_R_tot['{}-{}'.format(setup, rec_length)] = plots.get_CI_mean(rel_deviation_R_tot)

"""Plotting"""
fig, (axes) = plt.subplots(1, 2, figsize=(10, 3.2))
# fig.set_size_inches(4, 3)
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

    ##### y-axis ####
    ax.set_ylim((-10, 10))

    ##### Unset Borders #####
    ax.spines['top'].set_bounds(0, 0)
    ax.spines['right'].set_bounds(0, 0)
    ax.set_xlabel(r'recording time $T_{\mathrm{rec}}$ (min)')
    # only plot labels and legend for left-hand side
    if j == 0:
        ax.set_ylabel(r'relative bias for $\hat{R}_{\mathrm{tot}}$ (\%)')
        ax.set_title('no cross-validation')
    else:
        ax.set_title('with cross-validation')
    ax.plot(rec_length_values, np.zeros(len(rec_lengths)), color='0')

    for setup in setups[:,j]:
        regularization_method = setup.split("_")[1]
        for i, rec_length in enumerate(rec_lengths):
            mean_rel_deviation_R_tot_val = mean_rel_deviation_R_tot['{}-{}'.format(
                setup, rec_length)]
            mean_CI_rel_deviation_R_tot_val = mean_CI_rel_deviation_R_tot['{}-{}'.format(
                setup, rec_length)]
            mean_CI_lo = mean_CI_rel_deviation_R_tot_val[0] - mean_rel_deviation_R_tot_val
            mean_CI_hi = mean_CI_rel_deviation_R_tot_val[1] - mean_rel_deviation_R_tot_val
            if regularization_method == 'bbc':
                if j+i == 0:
                    ax.errorbar(x=[rec_length_values[i]], y=[mean_rel_deviation_R_tot_val], yerr=[[-mean_CI_lo], [mean_CI_hi]],
                                color=main_red, marker='d', markersize=5., capsize = 3.0, label = r'BBC, $d_{\mathrm{max}}=25$')
                else:
                    ax.errorbar(x=[rec_length_values[i]], y=[mean_rel_deviation_R_tot_val], yerr=[[-mean_CI_lo], [mean_CI_hi]],
                                color=main_red, marker='d', markersize=5., capsize = 3.0)
            else:
                if j+i == 0:
                    ax.errorbar(x=[rec_length_values[i]], y=[mean_rel_deviation_R_tot_val], yerr=[[-mean_CI_lo], [mean_CI_hi]],
                                color=main_blue, marker='d', markersize=5., capsize = 3.0, label = r'Shuffling, $d_{\mathrm{max}}=25$')
                else:
                    ax.errorbar(x=[rec_length_values[i]], y=[mean_rel_deviation_R_tot_val], yerr=[[-mean_CI_lo], [mean_CI_hi]],
                                color=main_blue, marker='d', markersize=5., capsize = 3.0)
    if j == 0:
        ax.legend(loc=(0.1, 0.03), frameon=False)

fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

plt.savefig('{}/S1Fig_{}.pdf'.format(PLOTTING_DIR, recorded_system),
            format="pdf", bbox_inches='tight')
plt.show()
plt.close()
