"""Functions"""
import matplotlib
from matplotlib import rc
import seaborn.apionly as sns
import pylab as plt
import numpy as np
from sys import exit, stderr, argv, path, modules
from os.path import isfile, isdir, realpath, dirname, exists
# import plotutils

PLOTTING_DIR = dirname(realpath(__file__))
CODE_DIR = '{}/../..'.format(PLOTTING_DIR)
path.insert(1, '{}/src'.format(CODE_DIR))
use_settings_path = False

if 'hde_glm' not in modules:
    import hde_glm as glm
    import hde_utils as utl
    import hde_plotutils as plots

"""Parameters and Settings"""
recorded_system = 'culture'
rec_length = '90min'
DATA_DIR = '{}/data/culture/'.format(CODE_DIR)
validNeurons = np.load('{}validNeurons.npy'.format(DATA_DIR))
T_0 = 0.00997

"""Plotting"""
# Font
rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '15.0'
matplotlib.rcParams['xtick.labelsize'] = '15'
matplotlib.rcParams['ytick.labelsize'] = '15'
matplotlib.rcParams['legend.fontsize'] = '15'
matplotlib.rcParams['axes.linewidth'] = 0.6

# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]
soft_red = sns.color_palette("RdBu_r", 15)[10]
soft_blue = sns.color_palette("RdBu_r", 15)[4]
violet = sns.cubehelix_palette(8)[4]
green = sns.cubehelix_palette(8, start=.5, rot=-.75)[3]

fig, axes = plt.subplots(8, 6, figsize=(14., 12.5))

# Sort neurons, put neurons with max_val > 0.2 and max_val <0.3 in a separate group

smallR = []
mediumR = []
highR = []
veryhighR = []
setup = 'full_shuffling'
for neuron_index, neuron in enumerate(validNeurons):
    ANALYSIS_DIR, analysis_num_str, R_tot, T_D, T, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = 'shuffling', use_settings_path = use_settings_path)
    max_val = np.amax(R)
    if max_val > 0.2:
        if max_val >0.3:
            if max_val > 0.405:
                veryhighR += [neuron_index]
            else:
                highR += [neuron_index]
        else:
            mediumR+= [neuron_index]
    else:
        smallR += [neuron_index]

index_small_to_medium = len(smallR)
index_medium_to_high = len(smallR)+len(mediumR)
index_high_to_veryhigh = len(smallR)+len(mediumR)+len(highR)

for k, neuron_index in enumerate(np.append(np.append(np.append(smallR, mediumR),highR),veryhighR)):

    ax = axes[int(k/6)][k%6]

    """Load data full"""
    setup = 'full_bbc'
    ANALYSIS_DIR, analysis_num_str, R_tot_bbc, T_D_bbc, T_bbc, R_bbc, R_bbc_CI_lo, R_bbc_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = 'bbc', use_settings_path = use_settings_path)
    R_tot_bbc, T_D_index_bbc, max_valid_index_bbc = plots.get_R_tot(T_bbc, R_bbc, R_bbc_CI_lo)
    dR_bbc = plots.get_dR(T_bbc ,R_bbc ,R_tot_bbc)
    tau_R_bbc = plots.get_T_avg(T_bbc, dR_bbc, T_0)
    # Get R_tot_glm for T_D
    R_tot_glm = plots.load_analysis_results_glm(ANALYSIS_DIR, analysis_num_str)

    setup = 'full_shuffling'
    ANALYSIS_DIR, analysis_num_str,R_tot_shuffling, T_D_shuffling, T, R_shuffling, R_shuffling_CI_lo, R_shuffling_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = 'shuffling', use_settings_path = use_settings_path)
    R_tot_shuffling, T_D_index_shuffling, max_valid_index_shuffling = plots.get_R_tot(T, R_shuffling, R_shuffling_CI_lo)
    dR_shuffling = plots.get_dR(T ,R_shuffling ,R_tot_shuffling)
    tau_R_shuffling = plots.get_T_avg(T, dR_shuffling, T_0)

    """Load data five bins"""
    setup = 'fivebins'
    ANALYSIS_DIR, analysis_num_str,R_tot_fivebins, T_D_fivebins, T, R_fivebins, R_fivebins_CI_lo, R_fivebins_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = 'shuffling', use_settings_path = use_settings_path)
    R_tot_fivebins, T_D_index_fivebins, max_valid_index_fivebins = plots.get_R_tot(T, R_fivebins, R_fivebins_CI_lo)
    dR_fivebins = plots.get_dR(T ,R_fivebins ,R_tot_fivebins)
    tau_R_fivebins = plots.get_T_avg(T, dR_fivebins, T_0)
    """Load data onebins"""
    setup = 'onebin'
    ANALYSIS_DIR, analysis_num_str,R_tot_onebin, T_D_onebin, T, R_onebin, R_onebin_CI_lo, R_onebin_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = 'shuffling', use_settings_path = use_settings_path)
    R_tot_onebin, T_D_index_onebin, max_valid_index_onebin = plots.get_R_tot(T, R_onebin, R_onebin_CI_lo)
    dR_onebin = plots.get_dR(T ,R_onebin ,R_tot_onebin)
    tau_R_onebin = plots.get_T_avg(T, dR_onebin, T_0)
    ax.set_xscale('log')
    x_min = 0.005
    x_max = 5.
    ax.set_xlim((0.005, 5.))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.spines['bottom'].set_bounds(0.005, 5.)
    ax.set_xticks(np.array([0.01, 0.1, 1.0]))
    ax.set_xticklabels(
        [r'$10$', r'$100$', r'$1000$'], rotation='horizontal')

    ##### y-axis ####
    ##### Unset Borders #####
    ax.spines['top'].set_bounds(0, 0)
    ax.spines['right'].set_bounds(0, 0)
    max_val = np.amax(R_shuffling)
    if max_val > 0.2:
        if max_val > 0.3:
            if max_val > 0.405:
                yrange = 0.41
                ymin = 0.1
                ax.set_ylim((0.1, .51))
                ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
                ax.spines['left'].set_bounds(.1, 0.5)
            else:
                yrange = 0.3
                ymin = 0.1
                ax.set_ylim((0.1, .4))
                ax.set_yticks([0.1, 0.2, 0.3, 0.4])
                ax.spines['left'].set_bounds(.1, 0.4)
        else:
            yrange = 0.3
            ymin = 0.0
            ax.set_ylim((0.0, .3))
            ax.set_yticks([0.0, 0.1, 0.2, 0.3])
            ax.spines['left'].set_bounds(.0, 0.3)
    else:
        yrange = 0.2
        ymin = 0.0
        ax.set_ylim((0.0, .2))
        ax.set_yticks([0.0, 0.1, 0.2])
        ax.spines['left'].set_bounds(.0, 0.2)

    ##### Unset Borders #####
    ax.spines['top'].set_bounds(0, 0)
    ax.spines['right'].set_bounds(0, 0)

    """BBC"""
    ax.plot(T_bbc, R_bbc, linewidth=1.2, color=main_red,
             label=r'BBC, $d_{\mathrm{max}}=20$', zorder=10)
    ax.fill_between(T_bbc, R_bbc_CI_lo, R_bbc_CI_hi,
                    facecolor=main_red, zorder= 10, alpha=0.3)

    # Rtot indicators
    x = (np.log10(T_D_bbc) - np.log10(x_min)) / \
        (np.log10(x_max) - np.log10(x_min))
    ax.axhline(y=R_tot_bbc, xmax=x, color=main_red,
                linewidth=0.5, linestyle='--')
    ax.plot([x_min], [R_tot_bbc], marker='d',markersize = 5., color=main_red,
             zorder=8)
    ax.plot([T_D_bbc], [R_tot_bbc], marker='|', markersize = 6., color=main_red,
             zorder=14)
    ax.plot([T_bbc[max_valid_index_bbc-1]], [R_tot_bbc], marker='|', markersize = 6., color=main_red,
             zorder=14)
    ax.plot(T_bbc[T_D_index_bbc:max_valid_index_bbc], np.zeros(max_valid_index_bbc-T_D_index_bbc)+R_tot_bbc, color = main_red, linewidth=1.5, linestyle='--')
    # tau_R indicators
    ax.plot([tau_R_bbc], [ymin], marker='d', markersize = 5., color=main_red,
             zorder=8)
    ax.axvline(x=tau_R_bbc, ymax=(R_tot_bbc - ymin) / yrange, color=main_red,
                linewidth=0.5, linestyle='--')

    if k == 0:
        ax.text(0.007, R_tot_bbc + 0.04 *
                 R_tot_bbc, r'$R_{\mathrm{tot}}$')
        ax.text(tau_R_bbc + 0.7 * tau_R_bbc, ymin + .005, r'$\tau_R$')


    """Shuffling"""
    ax.plot(T, R_shuffling, linewidth=1.2, color=main_blue,
             label=r'Shuffling, $d_{\mathrm{max}}=20$', zorder=3)
    ax.fill_between(T, R_shuffling_CI_lo, R_shuffling_CI_hi,
                facecolor=main_blue, zorder= 8, alpha=0.3)
    # Rtot indicators
    x = (np.log10(T_D_shuffling) - np.log10(x_min)) / \
        (np.log10(x_max) - np.log10(x_min))
    ax.axhline(y=R_tot_shuffling, xmax=x, color=main_blue,
                linewidth=0.5, linestyle='--')
    ax.plot([x_min], [R_tot_shuffling], marker='d',markersize = 5., color=main_blue,
             zorder=8)
    ax.plot(T[T_D_index_shuffling:max_valid_index_shuffling], np.zeros(max_valid_index_shuffling-T_D_index_shuffling)+R_tot_shuffling, color = main_blue, linewidth=1.5, linestyle='--')
    ax.plot([T_D_shuffling], [R_tot_shuffling], marker='|', markersize = 6., color=main_blue,
             zorder=13)
    ax.plot([T[max_valid_index_shuffling-1]], [R_tot_shuffling], marker='|', markersize = 6., color=main_blue, zorder=14)
    # tau_R indicators
    ax.axvline(x=tau_R_shuffling, ymax=(R_tot_shuffling - ymin) / yrange, color=main_blue,
                linewidth=0.5, linestyle='--')
    ax.plot([tau_R_shuffling], [ymin], marker='d', markersize = 5., color=main_blue,
             zorder=8)

    """Fivebins"""
    ax.plot(T, R_fivebins, linewidth=1.2, color=green,
             label=r'Shuffling, $d_{\mathrm{max}}=5$', zorder=3)
    ax.fill_between(T, R_fivebins_CI_lo, R_fivebins_CI_hi,
                    facecolor=green, zorder= 10, alpha=0.3)
    # Rtot indicators
    x = (np.log10(T_D_fivebins) - np.log10(x_min)) / \
        (np.log10(x_max) - np.log10(x_min))
    ax.axhline(y=R_tot_fivebins, xmax=x, color=green,
                linewidth=0.5, linestyle='--')
    ax.plot([x_min], [R_tot_fivebins], marker='d',markersize = 5., color=green,
             zorder=8)
    ax.plot(T[T_D_index_fivebins:max_valid_index_fivebins], np.zeros(max_valid_index_fivebins-T_D_index_fivebins)+R_tot_fivebins, color = green, linewidth=1.5, linestyle='--')
    ax.plot([T_D_fivebins], [R_tot_fivebins], marker='|', markersize = 6., color=green,
             zorder=12)
    ax.plot([T[max_valid_index_fivebins-1]], [R_tot_fivebins], marker='|', markersize = 6., color=green, zorder=14)
    # tau_R indicators
    ax.plot([tau_R_fivebins], [ymin], marker='d', markersize = 5., color=green,
             zorder=8)
    ax.axvline(x=tau_R_fivebins, ymax=(R_tot_fivebins - ymin) / yrange, color=green,
                linewidth=0.5, linestyle='--')

    """One bin"""
    ax.plot(T, R_onebin, linewidth=1.2, color='y',
             label=r'Shuffling, $d_{\mathrm{max}}=1$', zorder=3)
    ax.fill_between(T, R_onebin_CI_lo, R_onebin_CI_hi,
                    facecolor='y', zorder= 10, alpha=0.3)
    # Rtot indicators
    x = (np.log10(T_D_onebin) - np.log10(x_min)) / \
        (np.log10(x_max) - np.log10(x_min))
    ax.axhline(y=R_tot_onebin, xmax=x, color='y',
                linewidth=0.5, linestyle='--')
    ax.plot([x_min], [R_tot_onebin], marker='d',markersize = 5., color='y',
             zorder=8)
    ax.plot([T_D_onebin], [R_tot_onebin], marker='|', markersize = 6., color='y',
             zorder=8)
    ax.plot([T[max_valid_index_onebin-1]], [R_tot_onebin], marker='|', markersize = 6., color='y', zorder=14)
    ax.plot(T[T_D_index_onebin:max_valid_index_onebin], np.zeros(max_valid_index_onebin-T_D_index_onebin)+R_tot_onebin, color = 'y', linewidth=1.5, linestyle='--')
    # tau_R indicators
    ax.plot([tau_R_onebin], [ymin], marker='d', markersize = 5., color='y',
             zorder=8)
    ax.axvline(x=tau_R_onebin, ymax=(R_tot_onebin - ymin) / yrange, color='y',
                linewidth=0.5, linestyle='--')

    """GLM"""
    # Plot R_tot_glm
    ax.plot([T_D_bbc], [R_tot_glm], 's', color=violet, label=r'GLM, $d_{\mathrm{max}}=50$')
    x = (np.log10(T_D_bbc) - np.log10(x_min)) / \
        (np.log10(x_max) - np.log10(x_min))
    ax.plot([x_min], [R_tot_glm], marker='d',markersize = 5., color=violet,
             zorder=8)
    ax.axhline(y=R_tot_glm, xmax=x, color=violet,
                linewidth=0.5, linestyle='--')

    if not int(k/6) == 7:
        xlabels = [item.get_text() for item in ax.get_xticklabels()]
        empty_string_labels = ['']*len(xlabels)
        ax.set_xticklabels(empty_string_labels)
    if not k%6 == 0:
        if not k == index_small_to_medium:
            if not k == index_medium_to_high:
                if not k == index_high_to_veryhigh:
                    ylabels = [item.get_text() for item in ax.get_yticklabels()]
                    empty_string_labels = ['']*len(ylabels)
                    ax.set_yticklabels(empty_string_labels)
    if k == 0:
        ax.legend(loc=(-0.1, 1.1), frameon=False)

fig.text(0.5, - 0.01, r'past range $T$ (ms)', ha='center', va='center', fontsize = 17)
fig.text(-0.01, 0.5, r'history dependence $R(T)$', ha='center', va='center', rotation='vertical',  fontsize = 17)
fig.tight_layout(pad=1.0, w_pad=-1.0, h_pad=1.0)
plt.savefig('{}/S7Fig.pdf'.format(PLOTTING_DIR),
            format="pdf", bbox_inches='tight')

plt.show()
plt.close()
