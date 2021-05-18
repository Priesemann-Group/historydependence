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
from scipy.optimize import bisect
import random

PLOTTING_DIR = dirname(realpath(__file__))
CODE_DIR = '{}/../..'.format(PLOTTING_DIR)
path.insert(1, '{}/src'.format(CODE_DIR))

if 'hde_glm' not in modules:
        import hde_glm as glm
        import hde_utils as utl
        import hde_plotutils as plots

def get_p_spike_cond_eval(past, m, h, l):
    l = len(past)
    past_activation = np.sum(m/l*past)
    # past_activation = m*np.dot(np.exp(-np.arange(1,l+1)),past)/np.sum(np.exp(-np.arange(1,l+1)))
    p_spike_cond = h + (1-h)*past_activation
    return p_spike_cond

# in the dynamics, the decision h or 1-h is already taken when we compute the internal activation
def get_p_spike_cond_dyn(past, m, l):
    l = len(past)
    past_activation = np.sum(m/l*past)
    # past_activation = m*np.dot(np.exp(-np.arange(1,l+1)),past)/np.sum(np.exp(-np.arange(1,l+1)))
    p_spike_cond = past_activation
    return p_spike_cond

def get_p_spike_cond_eval_dict(m, h, l):
    p_spike_cond = {}
    for i in range(2**l):
        past = get_binary_past_from_integer(i,l)
        p_spike_cond[i] = get_p_spike_cond_eval(past, m, h, l)
    return p_spike_cond

def get_p_spike_cond_dyn_dict(m, h, l):
    p_spike_cond = {}
    for i in range(2**l):
        past = get_binary_past_from_integer(i,l)
        p_spike_cond[i] = get_p_spike_cond_dyn(past, m, l)
    return p_spike_cond

def get_binary_past_from_integer(x, l):
    past = np.zeros(l).astype(int)
    for i in range(l):
        state = x%(2**(i+1))
        if state > 0:
            past[i] = 1
            x -= 2**i
    return past

def simulate_spiketrain(N_steps, m, h, l):
    # initiate arrays:
    p_spike_cond = get_p_spike_cond_dyn_dict(m, h, l)
    signature = np.array([2**i for i in range(l)])
    past_states = np.zeros(N_steps)
    spikes = np.random.choice([0,1], N_steps, p = [1-h, h])
    past = np.dot(np.array([np.roll(spikes,k) for k in np.arange(1,l+1)]).T,signature)
    past[:l] = 0
    for i in range(N_steps):
        if past[i] > 0:
            if spikes[i] == 0:
                if random.random() < p_spike_cond[past[i]]:
                    spikes[i] = 1
                    for k in range(l):
                        if i+k+1 < N_steps:
                            past[i+k+1] += 2**k
    return spikes, past

def get_h_first_order(p_spike, m):
    return p_spike*(1-m)/(1-m*p_spike)

def get_spike_entropy(p_spike):
    p_nospike = 1 - p_spike
    if p_nospike == 1.0:
        entropy = 0
    else:
        entropy = - p_spike * np.log2(p_spike) - p_nospike * np.log2(p_nospike)
    return entropy

def R_first_order(m,h):
    p_spike = h/(1-m+h*m)
    p_nospike = 1 - p_spike
    spike_entropy = get_spike_entropy(p_spike)
    p_spike_past_spike = h + (1-h)*m
    p_spike_no_past_spike = h
    cond_entropy_past_spike = - p_spike_past_spike * np.log2(p_spike_past_spike) - (1-p_spike_past_spike) * np.log2(1-p_spike_past_spike)
    cond_entropy_no_past_spike = - p_spike_no_past_spike * np.log2(p_spike_no_past_spike) - (1-p_spike_no_past_spike) * np.log2(1-p_spike_no_past_spike)
    cond_entropy = p_spike * cond_entropy_past_spike + p_nospike * cond_entropy_no_past_spike
    I_pred = spike_entropy - cond_entropy
    R = I_pred/spike_entropy
    return R, I_pred

def get_R(spikes, past, p_spike_cond, l):
    N_steps = len(spikes)
    p_spike = np.sum(spikes)/N_steps
    p_nospike = 1 - p_spike
    spike_entropy = get_spike_entropy(p_spike)
    # historgram
    counts = np.histogram(past, bins = 2**l, range = (0,2**l))[0]
    cond_entropy = np.sum([counts[i]*get_spike_entropy(p_spike_cond[i]) for i in range(2**l)])/N_steps
    I_pred = spike_entropy - cond_entropy
    R = I_pred/spike_entropy
    return R, I_pred

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

def get_auto_correlation_time(spikes, min_steps, max_steps, bin_size_ms):
    rk = mre.coefficients(spikes, dt=bin_size_ms, steps = (min_steps, max_steps))
    T = (rk.steps-rk.steps[0]+1)*bin_size_ms
    fit = mre.fit(rk, steps = (min_steps,max_steps),fitfunc = mre.f_exponential)
    tau_C = fit.tau
    C_raw = rk.coefficients
    range = int(6*tau_C)
    tau_C_int = plots.get_T_avg(T[:range+1], C_raw[:range+1], 0)
    print(tau_C, tau_C_int)
    return tau_C_int, rk, T, fit

def get_lagged_MI(spikes, t):
    N = len(spikes)
    p_spike = np.sum(spikes)/N
    p_nospike = 1 - p_spike
    spike_entropy = get_spike_entropy(p_spike)
    # the t first bins do not have a past
    N -= t
    past = np.roll(spikes,t)
    counts_joint = np.histogram(past[t:] + spikes[t:] * 2, bins = 4, range = (0,4))[0]
    joint_entropy = np.sum(-counts_joint/N*np.log2(counts_joint/N))
    lagged_MI = 2*spike_entropy - joint_entropy
    return lagged_MI/spike_entropy


def get_tau_lagged_MI(spikes, max_steps):
    T_lagged_MI = np.arange(1,max_steps)
    lagged_MI_arr = []
    for t in T_lagged_MI:
        lagged_MI = get_lagged_MI(spikes, t)
        lagged_MI_arr += [lagged_MI]
    # add 0.5 because get_T_avg averags points in the center of bins, but here the smallest time step is the bin size so we want to average at the edges
    tau_lagged_MI = plots.get_T_avg(T_lagged_MI, np.array(lagged_MI_arr), 0) + 0.5
    return tau_lagged_MI, lagged_MI_arr, T_lagged_MI

def get_tau_R(spikes, past, l, R_tot):
    T = np.arange(1,l+1)
    R_arr = []
    for t in T:
        R_plugin = get_R_plugin(spikes, past, t)[0]
        R_arr += [R_plugin]
    dR_arr = plots.get_dR(T,R_arr,R_tot)
    # add 0.5 because get_T_avg averags points in the center of bins, but here the smallest time step is the bin size so we want to average at the edges
    tau_R = plots.get_T_avg(T, dR_arr, 0) + 0.5
    return tau_R, R_arr, dR_arr, T

"""Parameters"""
t_bin = 0.005
N_simulation_steps = 10000000
target_rate = 5 # in Hz
min_rate = 0.5 # in Hz
max_rate = 10 # in Hz
p_spike_target = target_rate * t_bin
p_spike_min = min_rate * t_bin
p_spike_max = max_rate * t_bin
min_steps_autocorrelation = 1
max_steps_autocorrelation = 150
bin_size_ms = 1.

# m has to be smaller than one: m<1
m_baseline = 0.8
l_max = 10

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

m_list = np.load('{}/analysis/binary_AR/m_list.npy'.format(CODE_DIR))
h_list = np.load('{}/analysis/binary_AR/h_list.npy'.format(CODE_DIR))
l_list = np.load('{}/analysis/binary_AR/l_list.npy'.format(CODE_DIR))

# PLOTTING R_tot, tau_R, tau_C, tau_L versus m for fixed l=1, and fixed rate = 5Hz
if argv[1] == 'vs_m':
    R_tot_list = []
    tau_R_list = []
    tau_lagged_MI_list = []
    tau_C_list = []
    l = 1
    m_list = np.linspace(0.5,0.95,15)
    R_tot_list = np.load('{}/analysis/binary_AR/R_tot_vs_m.npy'.format(CODE_DIR))
    tau_R_list = np.load('{}/analysis/binary_AR/tau_R_vs_m.npy'.format(CODE_DIR))
    tau_lagged_MI_list = np.load('{}/analysis/binary_AR/tau_lagged_MI_vs_m.npy'.format(CODE_DIR))
    tau_C_list = np.load('{}/analysis/binary_AR/tau_C_vs_m.npy'.format(CODE_DIR))
    fig, ((ax1,ax2,ax3)) = plt.subplots(nrows = 1, ncols = 3, figsize=(6.8, 1.8))
    # Plotting lagged MI
    for ax in (ax1,ax2,ax3):
        ax.spines['top'].set_bounds(0, 0)
        ax.spines['right'].set_bounds(0, 0)
        ax.set_xticks([0.5, 0.8])
    ax3.set_ylim([0,1])
    ax3.set_ylabel(r'\begin{center}total history  \\ dependence \end{center}')
    ax3.plot(m_list, R_tot_list, color = main_blue)
    # ax2.set_ylim([0,10])
    ax1.set_ylabel(r'\begin{center}generalized \\ timescale \end{center}')
    ax1.plot(m_list, tau_C_list, label = r'$\tau_C$', color = green)
    ax1.plot(m_list, tau_lagged_MI_list, label = r'$\tau_{L}$', color = 'orange')
    ax1.set_yticks([0,10,20])
    ax1.set_ylim([0,20])
    ax1.legend(loc = ((0.0,0.4)), frameon=False, fontsize = 14)
    ax2.plot(m_list, tau_R_list, label = r'$\tau_R$', color = main_blue)
    ax2.set_yticks([0,1,2])
    ax2.set_ylim([0,2.3])
    ax2.legend(loc = ((0.0,0.8)), frameon=False, fontsize = 14)
    fig.text(0.53,  0.005, r'coupling strength $m$', ha='center', va='center', fontsize = 15)
    # fig.tight_layout(pad=1.0, w_pad=-1.0, h_pad=1.0)
    fig.tight_layout()
    plt.savefig('%s/measures_%s.pdf'%(PLOTTING_DIR, argv[1]),
                format="pdf", bbox_inches='tight')
    plt.show()
    plt.close()

# PLOTTING R_tot, tau_R, tau_C, tau_L versus l for fixed R_tot, and fixed rate = 5Hz
if argv[1] == 'vs_l':
    R_tot_list = np.load('{}/analysis/binary_AR/R_tot_vs_l.npy'.format(CODE_DIR))
    tau_R_list = np.load('{}/analysis/binary_AR/tau_R_vs_l.npy'.format(CODE_DIR))
    tau_lagged_MI_list = np.load('{}/analysis/binary_AR/tau_lagged_MI_vs_l.npy'.format(CODE_DIR))
    tau_C_list = np.load('{}/analysis/binary_AR/tau_C_vs_l.npy'.format(CODE_DIR))
    fig, ((ax1,ax2,ax3)) = plt.subplots(nrows = 1, ncols = 3, figsize=(6.8, 1.8))
    # Plotting lagged MI
    for ax in (ax1,ax2, ax3):
        ax.spines['top'].set_bounds(0, 0)
        ax.spines['right'].set_bounds(0, 0)
    ax3.set_ylim([0,1])
    ax3.set_ylabel(r'\begin{center}total history  \\ dependence \end{center}')
    ax3.plot(l_list, R_tot_list, color = main_blue)
    ax1.set_ylabel(r'\begin{center}generalized \\ timescale \end{center}')

    ax1.plot(l_list, tau_C_list, label = r'$\tau_C$', color = green)
    ax1.plot(l_list, tau_lagged_MI_list, label = r'$\tau_{L}$', color = 'orange')
    ax1.set_yticks([0,25,50])
    ax1.set_ylim([0,55])
    ax2.plot(l_list, tau_R_list, color = main_blue, label = r'$\tau_R$')
    ax2.set_yticks([0,1,2])
    ax2.set_ylim([0,2.3])
    # ax2.legend(loc = ((0.0,0.62)), frameon=False, fontsize = 14)
    fig.text(0.53,  0.01, r'temporal depth $l$', ha='center', va='center', fontsize = 15)
    fig.tight_layout()
    plt.savefig('%s/measures_%s.pdf'%(PLOTTING_DIR, argv[1]),
                format="pdf", bbox_inches='tight')
    plt.show()
    plt.close()

# PLOTTING R_tot and I_pred vs h for fixed m = 0.8
if argv[1] == 'vs_h':
    m = m_baseline
    l = 1
    R_tot_list = []
    I_pred_list = []
    h_list = []
    for p_spike in np.linspace(p_spike_min,p_spike_max, 15):
        h = p_spike*(1-m)/(1-m*p_spike)
        R_tot, I_pred = R_first_order(m,h)
        R_tot_list += [R_tot]
        I_pred_list += [I_pred]
        h_list += [h]
    R_tot_list =np.array(R_tot_list)
    I_pred_list =np.array(I_pred_list)
    fig, ((ax1,ax2)) = plt.subplots(1, 2 , figsize=(4.8, 1.8))
    # Plotting lagged MI
    for ax in (ax1,ax2):
        ax.spines['top'].set_bounds(0, 0)
        ax.spines['right'].set_bounds(0, 0)
        ax.set_xticks([5,10])
    h_list = np.array(h_list)*1000
    ax1.text(8, 0.01, r'$\times 10^{-3}$')
    ax2.text(8, 0.05, r'$\times 10^{-3}$')
    ax1.set_ylabel(r'\begin{center}total mutual \\ information \end{center}')
    ax1.plot(h_list, I_pred_list, color = main_blue)
    ax1.set_ylim([0,0.2])
    ax2.set_ylim([0,1])
    ax2.set_ylabel(r'\begin{center}total history \\ dependence \end{center}')
    ax2.plot(h_list, R_tot_list, color = main_blue)
    fig.text(0.53,  0.03, r'input activation probability $h$', ha='center', va='center', fontsize = 15)
    fig.tight_layout()
    plt.savefig('%s/measures_%s.pdf'%(PLOTTING_DIR, argv[1]),
                format="pdf", bbox_inches='tight')
    plt.show()
    plt.close()

# PLOTTING different measures versus d for m = 0.8 and l=1/l=5, and rate = 5 Hz
if argv[1] == 'vs_d':
    if argv[2] == 'l=1':
        l = 1
    if argv[2] == 'l=5':
        l = 5
    m = m_list[l-1]
    h = h_list[l-1]
    fig, ((ax1,ax2)) = plt.subplots(2, 1 , figsize=(2.5, 4), sharex=True)
    # fig.set_title(r'$l=%d$'%l)
    for ax in (ax1,ax2):
        ax.spines['top'].set_bounds(0, 0)
        ax.spines['right'].set_bounds(0, 0)
    p_spike_cond = get_p_spike_cond_eval_dict(m_baseline, h, l)
    spikes, past = simulate_spiketrain(N_simulation_steps, m_baseline, h, l)
    # get R measures
    R_tot = get_R(spikes, past, p_spike_cond, l)[0]
    tau_R, R_arr, dR_arr, T_R = get_tau_R(spikes, past, l, R_tot)
    # get autocorrelation measures
    tau_C, rk, T_C, fit = get_auto_correlation_time(spikes, min_steps_autocorrelation, max_steps_autocorrelation, bin_size_ms)
    # get lagged MI measures
    tau_lagged_MI, lagged_MI_arr, T_lagged_MI = get_tau_lagged_MI(spikes, int(tau_C)*5)
    print(l, tau_C, tau_lagged_MI, tau_R)

    # Plotting auto-correlation
    ax1.set_title(r'$l=%d$'%l)
    ax1.set_xlim([0,20])
    ax1.bar(T_C, rk.coefficients, color = green, alpha = 0.6, zorder = 1)
    ax1.axvline(x=tau_C, ymax=1.0,
                linewidth=2, linestyle='--',color = green, zorder = 3)
    ax1.text(tau_C+1, rk.coefficients[0]*0.90, r'$\tau_C$')
    if l == 1:
        ax1.plot(T_C, mre.f_exponential(rk.steps, tau_C, *fit.popt[1:]), color = '0.1', zorder = 2)
        ax1.text(tau_C+1, rk.coefficients[0]*.5, r'$f(d)\propto e^{-\frac{d}{\tau_C}}$', fontsize = 14)
    else:
        ax1.set_yticks([0,0.3])
        ax1.text(tau_C+2, rk.coefficients[0]*0.5, r'$C(T)$')
    ax1.set_ylabel('autocorrelation $C$')
    ax1.legend(loc = ((0.2,0.62)), frameon=False, fontsize = 14)
    # Plotting lagged MI
    ax2.set_xlim([0,20])
    ax2.set_xlabel('time lag $T$ (steps)')
    ax2.set_ylabel(r'\begin{center}information \\measures \end{center}')
    ax2.bar(T_R, dR_arr, color = main_blue, alpha = 0.7, zorder = 2)
    ax2.bar(T_lagged_MI, lagged_MI_arr, color = 'orange', alpha = 0.4, zorder = 1)
    ax2.axvline(x=tau_R, ymax=1.0,
                linewidth=2, linestyle='--',color = main_blue, zorder = 3)
    ax2.text(tau_R-1, dR_arr[0]*1.1, r'$\tau_R$')
    ax2.axvline(x=tau_lagged_MI, ymax=1.0 ,
                linewidth=2, linestyle='--', color = 'orange',  zorder = 3)
    ax2.text(tau_lagged_MI+1, dR_arr[0]*1.1, r'$\tau_{L}$')
    if l==5:
        ax2.text(l-1.5, dR_arr[-1]*2.5, r'$\Delta R(T)$')
        ax2.text(tau_lagged_MI+3, dR_arr[0]*0.46, r'$L(T)/H$')
    fig.tight_layout(pad=1.0, w_pad=-1.0, h_pad=1.0)

    plt.savefig('%s/measures_%s_l=%d.pdf'%(PLOTTING_DIR, argv[1],l),
                format="pdf", bbox_inches='tight')
    plt.show()
    plt.close()
