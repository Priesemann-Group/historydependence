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

CODE_DIR = '{}/..'.format(dirname(realpath(__file__)))
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
    # rk_offset = fit.popt[2]
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

def save_m_and_h_lists(l_max, m_baseline, p_spike_target, adaptation_rate_m, adaptation_rate_h, N_simulation_steps, N_adaptation_steps, annealing_factor):
    nu_m = adaptation_rate_m # adaptation rate for m
    nu_h = adaptation_rate_h # adaptation rate for h
    h_baseline = p_spike_target*(1-m_baseline)/(1-m_baseline*p_spike_target)
    l_list = np.arange(1,l_max+1)
    R_target = R_first_order(m_baseline,h_baseline)[0]
    p_spike_list = [p_spike_target]
    R_list = [R_target]
    m_list = [m_baseline]
    h_list = [h_baseline]
    m = m_baseline
    h = h_baseline
    for l in l_list[1:]:
        N_steps = int(N_simulation_steps/2)
        nu_m = adaptation_rate_m # adaptation rate for m
        nu_h = adaptation_rate_h # adaptation rate for h
        for i in range(N_adaptation_steps):
            if i%5 == 0:
                p_spike_cond = get_p_spike_cond_eval_dict(m, h, l)
                spikes, past = simulate_spiketrain(N_steps, m, h, l)
                R = get_R(spikes, past, p_spike_cond, l)[0]
                # Adapt m
                m += nu_m * (R_target-R)
            # h_target = p_spike_target*(1-m)/(1-m*p_spike_target)
            p_spike_cond = get_p_spike_cond_eval_dict(m, h, l)
            spikes, past = simulate_spiketrain(N_steps, m, h, l)
            # Compute firing rate
            p_spike = np.sum(spikes)/N_steps
            h += nu_h * (p_spike_target - p_spike)
            print(l, p_spike/0.005, R)
            if i%10 ==0:
                nu_m = nu_m * annealing_factor
            if i%5 ==0:
                nu_h = nu_h * annealing_factor
            # print(rate, R)
        N_steps = N_simulation_steps
        p_spike_cond = get_p_spike_cond_eval_dict(m, h, l)
        spikes, past = simulate_spiketrain(N_steps, m, h, l)
        # Compute firing rate
        p_spike = np.sum(spikes)/N_steps
        R = get_R(spikes, past, p_spike_cond, l)[0]
        p_spike_list+=[p_spike]
        R_list += [R]
        m_list += [m]
        h_list += [h]
    plt.plot(l_list,R_list)
    plt.show()
    plt.close()
    plt.plot(l_list,p_spike_list)
    plt.show()
    plt.close()
    np.save('{}/analysis/binary_AR/l_list.npy'.format(CODE_DIR),l_list)
    np.save('{}/analysis/binary_AR/m_list.npy'.format(CODE_DIR),m_list)
    np.save('{}/analysis/binary_AR/h_list.npy'.format(CODE_DIR),h_list)
    np.save('{}/analysis/binary_AR/R_list.npy'.format(CODE_DIR),R_list)
    np.save('{}/analysis/binary_AR/p_spike_list.npy'.format(CODE_DIR),p_spike_list)

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

# Checking if analytical R and firing rate agree with the simulated ones.
if argv[1] == 'check':
    l = 1
    h = get_h_first_order(p_spike_target, m_baseline)
    p_spike_cond = get_p_spike_cond_eval_dict(m_baseline, h, l)
    spikes, past = simulate_spiketrain(N_simulation_steps, m_baseline, h, l)
    p_spike = np.sum(spikes)/N_simulation_steps
    rate = p_spike/t_bin
    R = get_R(spikes, past, p_spike_cond, l)[0]
    R_analytic =R_first_order(m_baseline,h)[0]
    print(rate, R, R_analytic)

# Adapt m and h to keep R and h fixed for different l
if argv[1] == 'adapt_m_and_h':
    N_adaptation_steps = 50
    adaptation_rate_m = 0.3
    learning_rate_h = 0.1
    annealing_factor = 0.85
    save_m_and_h_lists(l_max, m_baseline, p_spike_target, adaptation_rate_m, learning_rate_h, N_simulation_steps, N_adaptation_steps, annealing_factor)

# Compute R_tot, tau_R, tau_C, tau_L versus l for fixed R_tot, and fixed rate = 5Hz
if argv[1] == 'vs_l':
    R_tot_list = []
    tau_R_list = []
    tau_lagged_MI_list = []
    tau_C_list = []
    for i,l in enumerate(l_list):
        m = m_list[l-1]
        h = h_list[l-1]
        p_spike_cond = get_p_spike_cond_eval_dict(m, h, l)
        spikes, past = simulate_spiketrain(N_simulation_steps, m, h, l)
        R_tot = get_R(spikes, past, p_spike_cond, l)[0]
        tau_R = get_tau_R(spikes, past, l, R_tot)[0]
        # get autocorrelation measures
        tau_C = get_auto_correlation_time(spikes, min_steps_autocorrelation, max_steps_autocorrelation, bin_size_ms)[0]
        # get lagged MI measures
        tau_lagged_MI = get_tau_lagged_MI(spikes, int(tau_C)*5)[0]
        print(l, tau_C, tau_lagged_MI, tau_R)
        R_tot_list += [R_tot]
        tau_R_list += [tau_R]
        tau_lagged_MI_list += [tau_lagged_MI]
        tau_C_list += [tau_C]
    tau_R_list = np.array(tau_R_list)
    tau_C_list = np.array(tau_C_list)
    tau_lagged_MI_list = np.array(tau_lagged_MI_list)
    np.save('{}/analysis/binary_AR/tau_R_vs_l.npy'.format(CODE_DIR), tau_R_list)
    np.save('{}/analysis/binary_AR/tau_C_vs_l.npy'.format(CODE_DIR), tau_C_list)
    np.save('{}/analysis/binary_AR/tau_lagged_MI_vs_l.npy'.format(CODE_DIR), tau_lagged_MI_list)
    np.save('{}/analysis/binary_AR/R_tot_vs_l.npy'.format(CODE_DIR), R_tot_list)

# Compute R_tot, tau_R, tau_C, tau_L versus m for fixed l=1, and fixed rate = 5Hz
if argv[1] == 'vs_m':
    R_tot_list = []
    tau_R_list = []
    tau_lagged_MI_list = []
    tau_C_list = []
    l = 1
    m_list = np.linspace(0.5,0.95,15)
    for i,m in enumerate(m_list):
        h = p_spike_target*(1-m)/(1-m*p_spike_target)
        p_spike_cond = get_p_spike_cond_eval_dict(m, h, l)
        spikes, past = simulate_spiketrain(N_simulation_steps, m, h, l)
        R_tot = R_first_order(m,h)[0]
        tau_R = get_tau_R(spikes, past, l, R_tot)[0]
        # get autocorrelation measures
        tau_C = get_auto_correlation_time(spikes, min_steps_autocorrelation, max_steps_autocorrelation, bin_size_ms)[0]
        # get lagged MI measures
        tau_lagged_MI = get_tau_lagged_MI(spikes, int(tau_C)*5)[0]
        print(l, tau_C, tau_lagged_MI, tau_R)
        R_tot_list += [R_tot]
        tau_R_list += [tau_R]
        tau_lagged_MI_list += [tau_lagged_MI]
        tau_C_list += [tau_C]
    tau_R_list = np.array(tau_R_list)
    tau_C_list = np.array(tau_C_list)
    tau_lagged_MI_list = np.array(tau_lagged_MI_list)
    np.save('{}/analysis/binary_AR/tau_R_vs_m.npy'.format(CODE_DIR), tau_R_list)
    np.save('{}/analysis/binary_AR/tau_C_vs_m.npy'.format(CODE_DIR), tau_C_list)
    np.save('{}/analysis/binary_AR/tau_lagged_MI_vs_m.npy'.format(CODE_DIR), tau_lagged_MI_list)
    np.save('{}/analysis/binary_AR/R_tot_vs_m.npy'.format(CODE_DIR), R_tot_list)
