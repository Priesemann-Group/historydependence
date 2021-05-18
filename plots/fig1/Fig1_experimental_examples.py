
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
import mrestimator as mre

PLOTTING_DIR = dirname(realpath(__file__))
CODE_DIR = '{}/../..'.format(PLOTTING_DIR)
path.insert(1, '{}/src'.format(CODE_DIR))
use_settings_path = False

if 'hde_glm' not in modules:
        import hde_glm as glm
        import hde_utils as utl
        import hde_plotutils as plots

def get_spike_entropy(p_spike):
    p_nospike = 1 - p_spike
    if p_nospike == 1.0:
        entropy = 0
    else:
        entropy = - p_spike * np.log2(p_spike) - p_nospike * np.log2(p_nospike)
    return entropy

def get_R_plugin(spikes, past, l):
    N_steps = float(len(spikes))
    p_spike = np.sum(spikes)/N_steps
    p_nospike = 1 - p_spike
    spike_entropy = get_spike_entropy(p_spike)
    # How to preprocess past such that only the first l bins matter?
    past = past % 2**l
    counts_past =  np.histogram(past, bins = 2**l, range = (0,2**l))[0]
    counts_joint = np.histogram(past + spikes * 2**(l), bins = 2**(l+1), range = (0,2**(l+1)))[0]
    past_entropy = np.sum(-counts_past/N_steps*np.log2(counts_past/N_steps))
    joint_entropy = np.sum(-counts_joint/N_steps*np.log2(counts_joint/N_steps))
    I_pred = spike_entropy + past_entropy - joint_entropy
    R = I_pred/spike_entropy
    return R, I_pred

def get_lagged_MI(spiketimes, counts_from_spikes, bin_size, T_arr, Trec, median = False):
        N = len(counts_from_spikes)
        p_spike = np.sum(counts_from_spikes)/N
        spike_entropy = get_spike_entropy(p_spike)
        lagged_MI_arr = []
        # Add zero to compute delayed MI as dR, which ads a contribution from 0 to the first T in the list
        T_arr = np.append([0],T_arr)
        # compute medians
        for i,T in enumerate(T_arr[:-1]):
                past_counts = np.zeros(N)
                T_lo = T
                T_hi = T_arr[i+1]
                for t_spike in spiketimes:
                        if t_spike + T_hi < Trec:
                                max_bin = int((t_spike+T_hi)/bin_size)+1
                                min_bin = int((t_spike+T_lo)/bin_size)+1
                                for index in np.arange(min_bin,max_bin):
                                        past_counts[index] += 1
                median_past_counts = np.median(past_counts)
                if median == True:
                        past_counts[past_counts>median_past_counts] = 1
                        past_counts[past_counts<=median_past_counts] = 0
                N_bins_past = int(np.amax(past_counts)+1)
                N_bins_joint = int(np.amax(past_counts*2+counts_from_spikes)+1)
                counts_past =  np.histogram(past_counts, bins = N_bins_past, range = (0,N_bins_past))[0]
                counts_joint = np.histogram(past_counts*2 + counts_from_spikes, bins = N_bins_joint, range = (0,N_bins_joint))[0]
                past_entropy = np.sum(-counts_past/N*np.log2(counts_past/N))
                joint_entropy = np.sum(-counts_joint/N*np.log2(counts_joint/N))
                lagged_MI = spike_entropy + past_entropy - joint_entropy
                lagged_MI_arr += [lagged_MI]
        return np.array(lagged_MI_arr)/spike_entropy

def get_auto_correlation_time(counts_from_spikes, min_steps, max_steps, bin_size_ms):
    rk = mre.coefficients(counts_from_spikes, dt=bin_size_ms, steps = (min_steps, max_steps))
    fit = mre.fit(rk, steps = (min_steps,max_steps),fitfunc = mre.f_exponential_offset)
    tau_C = fit.tau
    return tau_C, fit

"""Parameters"""
recorded_system = 'V1'
setup = 'example_neuron'
regularization_method = 'shuffling'
rec_length = '90min'
# rec_length = '40min'
T_0 = 0.00997
T_0_ms = T_0 * 1000
t_0 = 100. - 10**(-4)
bin_size = 0.005
bin_size_ms = int(bin_size*1000)
# add by plus one because T_0 refers to lower bin edge, while the toolbox works with the upper edges of time bins (or steps)
min_step_autocorrelation = int(T_0_ms/bin_size_ms)+1
max_step_autocorrelation = 500
min_steps_plotting = 1
max_steps_plotting = 500

# Neuron '2-338'
neuron_index = 20

"""Load data R"""

ANALYSIS_DIR, analysis_num_str, R_tot, T_D, T_R, R, R_CI_lo, R_CI_hi = plots.load_analysis_results(
recorded_system, rec_length, neuron_index, setup, CODE_DIR, regularization_method = regularization_method, use_settings_path = use_settings_path)
R_tot, T_D_index, max_valid_index = plots.get_R_tot(T_R, R, R_CI_lo)
dR = plots.get_dR(T_R,R,R_tot)
T_R_ms= T_R*1000 # tranform measures to ms
tau_R = plots.get_T_avg(T_R_ms, dR, T_0_ms)
T_R_plotting = np.append(np.append([0],T_R_ms), [2000])
dR_plotting = np.append(dR, [0])
R_plotting = np.append(np.append([0],R), [R_tot])
print(R_tot)

"""Load and preprocess data"""
DATA_DIR = '{}/data/neuropixels/Waksman'.format(CODE_DIR)
validNeurons = np.load(
    '{}/validNeurons.npy'.format(DATA_DIR)).astype(int)
neuron = validNeurons[neuron_index]
print(neuron)
spiketimes = np.load('{}/V1/spks/spiketimes-{}-{}.npy'.format(DATA_DIR,neuron[0], neuron[1]))

# Add 5 seconds to make sure that only spikes with sufficient spiking history are considered
t_0 = spiketimes[0] + 5.
spiketimes = spiketimes - t_0
spiketimes = spiketimes[spiketimes > 0]
Trec = spiketimes[-1]

"""Compute measures"""
# Corr
counts_from_spikes = utl.get_binned_neuron_activity(spiketimes, bin_size)
tau_C, fit = get_auto_correlation_time(counts_from_spikes, min_step_autocorrelation, max_step_autocorrelation, bin_size_ms)
rk = mre.coefficients(counts_from_spikes, dt=bin_size_ms, steps = (min_steps_plotting, max_steps_plotting))
T_C = rk.steps*bin_size_ms
T_C_plotting = np.append([0],T_C)

# lagged mutualinformation for bins like R
T_L = np.arange(1,401)*bin_size
T_L_ms = T_L*1000
T_L_plotting = np.append([0],T_L_ms)
lagged_MI = get_lagged_MI(spiketimes, counts_from_spikes, bin_size, T_L, Trec, median = False)
lagged_MI_offset = np.mean(lagged_MI[150:])
tau_L = plots.get_T_avg(T_L_ms[:60], lagged_MI[:60]-lagged_MI_offset, T_0_ms)
print(tau_L)

"""Plotting"""
rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '15.0'
matplotlib.rcParams['xtick.labelsize'] = '15'
matplotlib.rcParams['ytick.labelsize'] = '15'
matplotlib.rcParams['legend.fontsize'] = '15'
matplotlib.rcParams['axes.linewidth'] = 0.6

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows= 2, ncols = 2 , figsize=(5.2, 4.2))

# fig.set_size_inches(4, 3)

# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]
soft_red = sns.color_palette("RdBu_r", 15)[10]
soft_blue = sns.color_palette("RdBu_r", 15)[4]
violet = sns.cubehelix_palette(8)[4]
green = sns.cubehelix_palette(8, start=.5, rot=-.75)[3]

##########################################
########## Simulated Conventional ########
##########################################

##### x-axis ####
# unset borders

for ax in [ax1,ax2,ax3,ax4]:
        ax.spines['top'].set_bounds(0, 0)
        ax.spines['right'].set_bounds(0, 0)
        ax.set_xscale('log')
        xmin = 6
        xmax = 2000
        ax.set_xlim((xmin,xmax))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.spines['bottom'].set_bounds(6, xmax)
        ax.set_xticks([10,100,1000])
        # ax.set_xticklabels([r'$T_0$'], rotation='horizontal')
        # ax.set_yticks([])

# ax1.set_xlabel(r'time lag $T$ (ms)')
fig.text(0.5,  0.48, r'time lag $T$ (ms)', ha='center', va='center', fontsize = 15)
fig.text(0.5,  0.03, r'past range $T$ (ms)', ha='center', va='center', fontsize = 15)
# ax3.set_xlabel(r'past range $T$ (ms)')

# Plot autocorrelation
ax1.set_ylabel('autocorrelation $C(T)$')
ax1.step(T_C_plotting[:-1], rk.coefficients, color = green, where = 'post')
# ax1.plot([xmin,xmax], [0,0], color = '0.2', linewidth = 1.2)
ax1.axvline(x=tau_C, ymax=1.03,
            linewidth=1.5, linestyle='--',color = '0.4', zorder = 3)

# Plot lagged mutualinfo (L)
ax2.set_ylabel(r'lagged MI $L(T)/H$')
# ax2.set_ylim([])
# ax2.plot(T_auto_MI, auto_MI,  color = green)
ax2.step(T_L_plotting[:-1], lagged_MI,  color = green, where = 'post')
# ax2.plot([xmin,xmax], [0,0], color = '0.2', linewidth = 1.2)
ax2.axvline(x=tau_L, ymax=1.03,
            linewidth=1.5, linestyle='--',color = '0.4', zorder = 3)

# Plot R
ax3.set_ylabel(r'$R(T)$')
ax3.plot(T_R_plotting, R_plotting, color = green)
# ax3.plot([xmin,xmax], [0,0], color = '0.2', linewidth = 1.2)
ax3.axhline(y=R_tot, xmax=1.0,
                    linewidth=1.5, linestyle='--',color = '0.4', zorder = 3)

# Plot Delta R
ax4.set_ylabel(r'gain $\Delta R(T)$')
#shift dR_arr_long, because values refer to upper bin edge, while we plot the lower
ax4.step(T_R_plotting[:-1], dR_plotting, color = green, where = 'post')
# ax4.step(np.append([0],T_ms[:-1]), lagged_MI,  color = green, where = 'post')
# ax4.plot([xmin,xmax], [0,0], color = '0.2', linewidth = 1.2)
ax4.axvline(x=tau_R, ymax=1.03,
            linewidth=1.5, linestyle='--',color = '0.4', zorder = 3)

ax2.set_ylim(ax4.get_ylim())
fig.tight_layout(pad = 1.1)
# Save fig
plt.savefig('{}/Fig1_experimental_examples.pdf'.format(PLOTTING_DIR, recorded_system),
            format="pdf", bbox_inches='tight')
plt.show()
plt.close()
