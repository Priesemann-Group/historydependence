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
import pickle

CODE_DIR = '{}/..'.format(dirname(realpath(__file__)))
path.insert(1, '{}/src'.format(CODE_DIR))
use_settings_path = False
data_path = '{}/data'.format(CODE_DIR)
analysis_path = '{}/analysis'.format(CODE_DIR)

if 'hde_glm' not in modules:
        import hde_glm as glm
        import hde_utils as utl
        import hde_plotutils as plots

def get_auto_correlation_time(counts_from_spikes, min_steps, max_steps, bin_size_ms):
    rk = mre.coefficients(counts_from_spikes, dt=bin_size_ms, steps = (min_steps, max_steps))
    fit = mre.fit(rk, steps = (min_steps,max_steps),fitfunc = mre.f_exponential_offset)
    tau_C = fit.tau
    # Computing the integrated timescale on offset corrected coefficients (does not work well because of huge fluctuations in autocorrelation)
    # rk_offset = fit.popt[2]
    # C_raw = rk.coefficients-rk
    # range = int(6*tau_C)
    # tau_C_int = plots.get_T_avg(T[:range+1], C_raw[:range+1], 0)
    return tau_C, fit

def get_spiking_stats(spiketimes):
    rate = len(spiketimes)/(spiketimes[-1]-spiketimes[0])
    ISIs = (spiketimes - np.roll(spiketimes,1))[1:]
    median_ISI = np.median(ISIs)
    mean_ISI = np.mean(ISIs)
    std_ISI = np.std(ISIs)
    CV = std_ISI/mean_ISI
    return rate, median_ISI, CV

"""Parameters"""
bin_size = 0.005 # sec
bin_size_ms = 5 # ms
T_0_ms = int(argv[1]) # ms
bin_size_ms = int(bin_size*1000)
min_step_autocorrelation = int(T_0_ms/bin_size_ms)+1
max_step_autocorrelation = 500
min_steps_plotting = 1
max_steps_plotting = 500

"""Load spiking data"""
# recorded_system = argv[1]
for recorded_system in ['retina', 'culture', 'CA1', 'V1']:
    DATA_DIR = '{}/{}'.format(data_path,recorded_system)
    if recorded_system == 'V1':
        DATA_DIR = '{}/neuropixels/Waksman/V1'.format(data_path)
    N_neurons_dict = {'CA1' : 28, 'retina': 111, 'culture': 48, 'V1': 142}
    N_neurons = N_neurons_dict[recorded_system]
    validNeurons = np.load(
        '{}/validNeurons.npy'.format(DATA_DIR)).astype(int)
    for neuron_index in range(N_neurons):
    # for neuron_index in [1,20,30]:
        neuron = validNeurons[neuron_index]
        print(neuron)
        if recorded_system == 'V1':
            spiketimes = np.load('{}/spks/spiketimes-{}-{}.npy'.format(DATA_DIR,neuron[0], neuron[1]))
        else:
            spiketimes = np.load(
                '{}/spks/spiketimes_neuron{}.npy'.format(DATA_DIR, neuron))

        # Add 5 seconds to make sure that only spikes with sufficient spiking history are considered
        t_0 = spiketimes[0] + 5.
        spiketimes = spiketimes - t_0
        spiketimes = spiketimes[spiketimes > 0]

        """Compute autocorrelation time, median ISI and CV"""
        counts_from_spikes = utl.get_binned_neuron_activity(spiketimes, bin_size)
        tau_C, fit = get_auto_correlation_time(counts_from_spikes, min_step_autocorrelation, max_step_autocorrelation, bin_size_ms)
        rate, median_ISI, CV = get_spiking_stats(spiketimes)
        print(rate, tau_C)
        stats_dict = {'rate':rate, 'medianISI': median_ISI, 'CV' : CV, 'autocorrelation_time': tau_C}

        # """Save result"""
        with open('%s/%s/stats_tbin_%dms/stats_neuron%d_T_0_%dms.pkl'%(analysis_path, recorded_system, bin_size_ms, neuron_index, T_0_ms), 'wb') as f:
            pickle.dump(stats_dict, f, pickle.HIGHEST_PROTOCOL)
