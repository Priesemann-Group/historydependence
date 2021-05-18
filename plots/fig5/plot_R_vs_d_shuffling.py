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
from scipy.optimize import bisect

PLOTTING_DIR = dirname(realpath(__file__))
CODE_DIR = '{}/../..'.format(PLOTTING_DIR)
path.insert(1, '{}/src'.format(CODE_DIR))
use_settings_path = False

if 'hde_glm' not in modules:
    import hde_glm as glm
    import hde_utils as utl
    import hde_plotutils as plots

recorded_system = 'glif_1s_kernel'
rec_length = '90min'

def get_std_vs_d(estimator, N, tau, ANALYSIS_DIR):
    sample_list = []
    for sample_index in range(N):
        estimator_vs_d = np.load('%s/%s_vs_d_tau%dms_%d.npy'%(ANALYSIS_DIR,estimator, int(tau*1000), sample_index), allow_pickle= True)
        sample_list += [estimator_vs_d]
    # print(estimator, np.mean(sample_list, axis = 0))
    return np.std(sample_list, axis = 0).astype(float)

"""Parameters"""
ANALYSIS_DIR = '{}/analysis/{}/analysis_R_vs_d'.format(CODE_DIR, recorded_system)
tau = 0.02
d_list = np.load('{}/embedding_number_of_bins_set.npy'.format(ANALYSIS_DIR)).astype(int)
N_d = len(d_list)

"""Load data """
sample_index = 0
R_plugin = np.load('%s/R_plugin_vs_d_tau%dms_%d.npy'%(ANALYSIS_DIR,int(tau*1000), sample_index))
R_nsb = np.load('%s/R_nsb_vs_d_tau%dms_%d.npy'%(ANALYSIS_DIR,int(tau*1000), sample_index), allow_pickle= True).astype(float)
bbc_term = np.load('%s/bbc_term_vs_d_tau%dms_%d.npy'%(ANALYSIS_DIR,int(tau*1000), sample_index), allow_pickle= True)
R_shuffling = np.load('%s/R_shuffling_vs_d_tau%dms_%d.npy'%(ANALYSIS_DIR,int(tau*1000), sample_index))
R_shuffling_correction = np.load('%s/R_shuffling_correction_vs_d_tau%dms_%d.npy'%(ANALYSIS_DIR,int(tau*1000), sample_index))

R_plugin_std = get_std_vs_d('R_plugin', 50, tau, ANALYSIS_DIR)
R_nsb_std = get_std_vs_d('R_nsb', 50, tau, ANALYSIS_DIR)
R_shuffling_std = get_std_vs_d('R_shuffling', 50, tau, ANALYSIS_DIR)
R_shuffling_correction_std = get_std_vs_d('R_shuffling_correction', 50, tau, ANALYSIS_DIR)

"""GLM benchmark"""
glm_csv_file_name = '{}/glm_benchmark_900min_tau20.csv'.format(
    ANALYSIS_DIR)
R_glm = pd.read_csv(glm_csv_file_name)['R_GLM']
R_tot = np.load('{}/analysis/{}/R_tot_900min.npy'.format(CODE_DIR, recorded_system))

bbc_tolerance = 0.05
R_bbc = R_nsb[bbc_term < bbc_tolerance]
d_opt_index = len(R_bbc) - 1

rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '14.0'
matplotlib.rcParams['xtick.labelsize'] = '14'
matplotlib.rcParams['ytick.labelsize'] = '14'
matplotlib.rcParams['legend.fontsize'] = '14'
matplotlib.rcParams['axes.linewidth'] = 0.6

fig, ((ax1)) = plt.subplots(1, 1, figsize=(3.7, 2.8))
# fig.set_size_inches(4, 3)

main_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]
main_green = sns.color_palette("BuGn_r", 15)[4]
# sns.palplot(sns.color_palette("BuGn_r", 15))  #visualize the color palette

##########################################
########## Plotting ########
##########################################

##### x-axis ####
ax1.set_xscale('log')
ax1.set_xlim((1, 60))
ax1.set_xticks(np.array([1, 10, 60]))
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.spines['bottom'].set_bounds(1, 60)
ax1.set_xlabel(r'embedding dimension $d$')

##### y-axis ####
ax1.set_ylabel(r'history dependence $R(\tau,d)$')
ax1.set_ylim((-0.0, .41))
ax1.set_yticks([0.0, 0.1, 0.2, 0.4])
ax1.spines['left'].set_bounds(.0, 0.4)


##### Unset Borders #####
ax1.spines['top'].set_bounds(0, 0)
ax1.spines['right'].set_bounds(0, 0)

ax1.plot(d_list, R_glm, linewidth=1.5, color='0.5', label='true', zorder=8)
ax1.plot(d_list[:-1], R_plugin, linewidth=1.5,
         linestyle='--',
         color=main_blue, label=r'ML')
ax1.fill_between(d_list[:-1], R_plugin - 2 * R_plugin_std,
                 R_plugin + 2 * R_plugin_std, facecolor=main_blue, alpha=0.3)

ax1.plot(d_list[:-1], R_shuffling_correction, '--',
         linewidth=1.5, color='y', label='est. bias', zorder=1)
ax1.fill_between(d_list[:-1], R_shuffling_correction - 2 * R_shuffling_correction_std,
                 R_shuffling_correction + 2 * R_shuffling_correction_std, facecolor='y', alpha=0.3)

ax1.plot(d_list[:-1], R_shuffling, label="Shuffling",
         linewidth=1.5, color=main_blue, zorder=9)
ax1.fill_between(d_list[:-1], R_shuffling - 2 * R_shuffling_std,
                 R_shuffling + 2 * R_shuffling_std, facecolor=main_blue, alpha=0.5)
R_tot_shuffling = np.amax(R_shuffling)
ax1.plot(d_list[:-1][np.where(R_shuffling == R_tot_shuffling)], [
         R_tot_shuffling], 'd', color=main_blue, zorder=9)
ax1.text(21, 0.04, r'\begin{center}Shuffling = \\ ML - est. bias \end{center}',
         color='0.35', ha='left', va='bottom')

ax1.legend(loc=(0.05, 0.45), frameon=False)

fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
plt.savefig('{}/Fig5A_R_vs_d_shuffling.pdf'.format(PLOTTING_DIR),
            format="pdf", bbox_inches='tight')

plt.show()
plt.close()
