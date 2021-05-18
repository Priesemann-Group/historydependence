
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

CODE_DIR = '{}/..'.format(dirname(realpath(__file__)))
path.insert(1, '{}/src'.format(CODE_DIR))
use_settings_path = False

if 'hde_glm' not in modules:
        import hde_utils as utl
        import hde_plotutils as plots

def get_spike_entropy(p_spike):
        p_nospike = 1 - p_spike
        if p_nospike == 1.0:
        entropy = 0
        else:
        entropy = - p_spike * np.log2(p_spike) - p_nospike * np.log2(p_nospike)
        return entropy

def get_lagged_MI(spiketimes, bin_size, T_arr, Trec, median = False):
        N = int(Trec/bin_size)
        counts_from_sptimes = np.zeros(N)
        for t_spike in spiketimes:
                if t_spike < Trec:
                        bin_index = int(t_spike/bin_size)
                        counts_from_sptimes[bin_index] =1
        p_spike = np.sum(counts_from_sptimes)/N
        spike_entropy = get_spike_entropy(p_spike)
        lagged_MI_arr = []
        T_arr = np.append([0],T_arr)
        for i,T in enumerate(T_arr[:-1]):
                past_counts = np.zeros(N)
                T_lo = T
                T_hi = T_arr[i+1]
                for t_spike in spiketimes:
                        if t_spike + T_lo < Trec:
                                max_bin = int((t_spike+T_hi)/bin_size)+1
                                min_bin = int((t_spike+T_lo)/bin_size)+1
                                max_bin = np.amin([max_bin,N])
                                for index in np.arange(min_bin,max_bin):
                                        past_counts[index] += 1
                median_past_counts = np.median(past_counts)
                if median == True:
                        past_counts[past_counts>median_past_counts] = 1
                        past_counts[past_counts<=median_past_counts] = 0
                N_bins_past = int(np.amax(past_counts)+1)
                N_bins_joint = int(np.amax(past_counts*2+counts_from_sptimes)+1)
                counts_past =  np.histogram(past_counts, bins = N_bins_past, range = (0,N_bins_past))[0]
                counts_joint = np.histogram(past_counts*2 + counts_from_sptimes, bins = N_bins_joint, range = (0,N_bins_joint))[0]
                past_entropy = np.sum(-counts_past/N*np.log2(counts_past/N))
                joint_entropy = np.sum(-counts_joint/N*np.log2(counts_joint/N))
                lagged_MI = spike_entropy + past_entropy - joint_entropy
                lagged_MI_arr += [lagged_MI]
        return np.array(lagged_MI_arr)/spike_entropy

"""Parameters"""
recorded_system = 'branching_process'
rec_length = '90min'
sample_index = 0
setup = 'full_shuffling'

tau = 198
bin_size = 0.004
bin_size_ms = int(bin_size*1000)

"""Load and preprocess data"""
DATA_DIR = '{}/data/{}'.format(CODE_DIR, recorded_system)
spiketimes = np.load('{}/spiketimes_tau{}ms_{}.npy'.format(DATA_DIR,int(tau),rec_length))
# same processing as in the HD toolbox
spiketimes = spiketimes - spiketimes[0]
Trec = spiketimes[-1] - spiketimes[0]
counts_from_sptimes = utl.get_binned_neuron_activity(spiketimes, bin_size)

"""Compute lagged mutual information"""
T_L = np.arange(1,251)*bin_size
lagged_MI = get_lagged_MI(spiketimes, bin_size, T_L, Trec, median = False)
np.save('{}/analysis/{}/lagged_MI.npy'.format(CODE_DIR, recorded_system), lagged_MI)
