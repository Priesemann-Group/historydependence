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
regularization_method = 'shuffling'
rec_length = '40min'
T_0 = 0.00997 # sec
neuron_index_list = [52, 104, 62]

for i, neuron_index in enumerate(neuron_index_list):
    panel = ['A','B','C'][i]
    # '2-338' : normal (in the new validNeurons script index 20)
    # '2-303': long-range (in the new validNeurons script index 1)
    # '2-357' : bursty (in the new validNeurons script index 30)

    ANALYSIS_DIR, analysis_num_str, R_tot, T_D, T, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(
        recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = regularization_method, use_settings_path = use_settings_path)
    R_tot, T_D_index, max_valid_index = plots.get_R_tot(T, R, R_CI_lo)
    dR = plots.get_dR(T,R,R_tot)
    T= T*1000 # tranform to ms
    T_D = T_D * 1000
    tau_R = plots.get_T_avg(T, dR, T_0)

    """Plot"""

    rc('text', usetex=True)
    matplotlib.rcParams['font.size'] = '16.0'
    matplotlib.rcParams['xtick.labelsize'] = '16'
    matplotlib.rcParams['ytick.labelsize'] = '16'
    matplotlib.rcParams['legend.fontsize'] = '16'
    matplotlib.rcParams['axes.linewidth'] = 0.6
    # Colors
    main_red = sns.color_palette("RdBu_r", 15)[12]
    main_blue = sns.color_palette("RdBu_r", 15)[1]
    soft_red = sns.color_palette("RdBu_r", 15)[10]
    soft_blue = sns.color_palette("RdBu_r", 15)[4]
    violet = sns.cubehelix_palette(8)[4]
    green = sns.cubehelix_palette(8, start=.5, rot=-.75)[3]

    ##### x-axis ####
    if i ==0:
        fig, ((ax)) = plt.subplots(nrows= 1, ncols = 1 , figsize=(3.8, 2.8))
    else:
        fig, ((ax)) = plt.subplots(nrows= 1, ncols = 1 , figsize=(3.5, 2.8))

    ax.set_xscale('log')
    x_min = 5
    x_max = 5000
    ax.set_xlim((5, 5000))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.spines['bottom'].set_bounds(5, 5000)
    ax.set_xticks([10, 100, 1000])
    if neuron_index == 104:
        ax.set_xlabel(r'past range $T$ (ms)')
    else:
        ax.set_xlabel(r'past range $T$ (ms)',alpha=0.0)

    ##### y-axis ####
    if neuron_index == 52:
        ax.set_ylabel(r'history dependence $R(T)$')
    else:
        ax.set_ylabel(r'history dependence $R(T)$',alpha=0.0)
    max_val = np.amax(R)
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

    if not neuron_index == 52:
        labels = [item.get_text() for item in ax.get_yticklabels()]
        empty_string_labels = ['']*len(labels)
        ax.set_yticklabels(empty_string_labels)

    ax.plot(T, R, linewidth=1.2, color=green,
             label='Shuffling', zorder=3)
    ax.fill_between(T, R_CI_lo, R_CI_hi, facecolor=green, alpha=0.3)


    # shuffled
    ax.plot([tau_R], [ymin], marker='d', markersize = 5., color=green,
             zorder=8)
    ax.plot([T_D], [R_tot], marker='|', markersize=7, color=green,
            zorder=8)
    ax.plot([T[max_valid_index-1]], [R_tot], marker='|', markersize=7, color=green,zorder=8)
    ax.axvline(x=tau_R, ymax=(R_tot - ymin) / yrange, color=green,
                linewidth=1.0, linestyle='--')
    x = (np.log10(T_D) - np.log10(x_min)) / \
        (np.log10(x_max) - np.log10(x_min))
    ax.axhline(y=R_tot, xmax=x, color=green,
                linewidth=1.0, linestyle='--')
    ax.plot([x_min], [R_tot], marker='d',markersize = 5., color=green,
             zorder=8)
    if i == 0:
        ax.text(7, R_tot + 0.06 *
                 R_tot, r'$R_{\mathrm{tot}}$')
        ax.text(tau_R + 0.2 * tau_R, .005, r'$\tau_R$')

    ax.plot(T[T_D_index:max_valid_index], np.zeros(max_valid_index-T_D_index)+R_tot, color = green,linestyle='--')

    fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

    plt.savefig('%s/Fig8%s_R_vs_T.pdf'%(PLOTTING_DIR,panel),
                format="pdf", bbox_inches='tight')

    plt.close()
