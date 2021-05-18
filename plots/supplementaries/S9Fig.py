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

"""Load data"""
recorded_system = 'V1'
setup = 'fivebins'
rec_length = '40min'
T_0 = 0.00997

DATA_DIR = '{}/data/neuropixels/Waksman/'.format(CODE_DIR)
validNeurons = np.load('{}validNeurons.npy'.format(DATA_DIR)).astype(int)
print(len(validNeurons))

"""Plot"""

rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '13.0'
matplotlib.rcParams['xtick.labelsize'] = '13'
matplotlib.rcParams['ytick.labelsize'] = '13'
matplotlib.rcParams['legend.fontsize'] = '13'
matplotlib.rcParams['axes.linewidth'] = 0.6
# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]
soft_red = sns.color_palette("RdBu_r", 15)[10]
soft_blue = sns.color_palette("RdBu_r", 15)[4]
violet = sns.cubehelix_palette(8)[4]
green = sns.cubehelix_palette(8, start=.5, rot=-.75)[3]

##### x-axis ####


fig, axes = plt.subplots(18, 8, figsize=(14., 19.5))

# Sort neurons, put neurons with max_val > 0.2 and max_val <0.3 in a separate group

normalR = []
highR = []
veryhighR = []
for neuron_index, neuron in enumerate(validNeurons):
    ANALYSIS_DIR, analysis_num_str, R_tot, T_D, T, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = 'shuffling', use_settings_path = use_settings_path)
    max_val = np.amax(R)
    if max_val > 0.2:
        if max_val >0.3:
            veryhighR += [neuron_index]
        else:
            highR+= [neuron_index]
    else:
        normalR += [neuron_index]

for k, neuron_index in enumerate(np.append(np.append(highR, normalR),veryhighR)):
    ANALYSIS_DIR, analysis_num_str, R_tot, T_D, T, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = 'shuffling', use_settings_path = use_settings_path)
    R_tot, T_D_index, max_valid_index = plots.get_R_tot(T, R, R_CI_lo)
    dR = plots.get_dR(T ,R ,R_tot)
    tau_R = plots.get_T_avg(T, dR, T_0)

    ax = axes[int(k/8)][k%8]
    # fig.set_size_inches(4, 3)
    ax.set_xscale('log')
    x_min = 0.005
    x_max = 5.
    ax.set_xlim((0.005, 5.))
    # ax.set_xticks(np.array([1, 10, 50]))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.spines['bottom'].set_bounds(0.005, 5.)
    ax.set_xticks(np.array([0.01, 0.1, 1.0]))
    ax.set_xticklabels(
        [r'$10$', r'$100$', r'$1000$'], rotation='horizontal')

    max_val = np.amax(R)
    if max_val > 0.2:
        if max_val > 0.3:
            if max_val > 0.405:
                yrange = 0.41
                ymin = 0.1
                ax.set_ylim((0.1, .55))
                ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
                ax.spines['left'].set_bounds(.1, 0.5)
            else:
                yrange = 0.3
                ymin = 0.1
                ax.set_ylim((0.1, .45))
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
    if not int(k/8) == 17:
        xlabels = [item.get_text() for item in ax.get_xticklabels()]
        empty_string_labels = ['']*len(xlabels)
        ax.set_xticklabels(empty_string_labels)
    if not k%8 == 0:
        if not k==141:
            ylabels = [item.get_text() for item in ax.get_yticklabels()]
            empty_string_labels = ['']*len(ylabels)
            ax.set_yticklabels(empty_string_labels)

    ##### Unset Borders #####
    ax.spines['top'].set_bounds(0, 0)
    ax.spines['right'].set_bounds(0, 0)


    ax.plot(T, R, linewidth=1.2, color=green,
             label='Shuffling', zorder=3)
    ax.fill_between(T, R_CI_lo, R_CI_hi, facecolor=green, alpha=0.3)

    # Rtot indicators
    x = (np.log10(T_D) - np.log10(x_min)) / \
        (np.log10(x_max) - np.log10(x_min))
    ax.axhline(y=R_tot, xmax=x, color=green,
                linewidth=0.5, linestyle='--')
    ax.plot([x_min], [R_tot], marker='d',markersize = 5., color=green,
             zorder=8)
    ax.plot([T_D], [R_tot], marker='|', markersize = 6., color=green,
             zorder=14)
    ax.plot([T[max_valid_index-1]], [R_tot], marker='|', markersize = 6., color=green,
             zorder=14)
    ax.plot(T[T_D_index:max_valid_index], np.zeros(max_valid_index-T_D_index)+R_tot, color = green, linewidth=1.5, linestyle='--')
    # tau_R indicators
    ax.plot([tau_R], [ymin], marker='d', markersize = 5., color=green,
             zorder=8)
    ax.axvline(x=tau_R, ymax=(R_tot - ymin) / yrange, color=green,
                linewidth=0.5, linestyle='--')
    if k == 1:
        ax.text(0.007, R_tot + 0.06 *
                 R_tot, r'$R_{\mathrm{tot}}$')
        ax.text(tau_R + 0.2 * tau_R, .005, r'$\tau_R$')

for j in np.arange(k+1,8*18):
    ax = axes[int(j/8)][j%8]
    ax.spines['top'].set_bounds(0, 0)
    ax.spines['right'].set_bounds(0, 0)
    ax.spines['bottom'].set_bounds(0, 0)
    ax.spines['left'].set_bounds(0, 0)
    ax.set_xticks([])
    ax.set_yticks([])


fig.text(0.5, - 0.01, r'past range $T$ (ms)', ha='center', va='center', fontsize = 17)
fig.text(-0.01, 0.5, r'history dependence $R(T)$', ha='center', va='center', rotation='vertical',  fontsize = 17)
fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

plt.savefig('%s/S9Fig.pdf'%(PLOTTING_DIR),
            format="pdf", bbox_inches='tight')

plt.show()
plt.close()
