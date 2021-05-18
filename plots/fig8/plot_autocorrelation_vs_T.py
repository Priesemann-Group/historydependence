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
T_0 = 0.01 # sec
neuron_list = [20, 1, 30]

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

PLOTTING_DIR = dirname(realpath(__file__))
CODE_DIR = '{}/../..'.format(PLOTTING_DIR)
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
    # Computing the generalized timescale on offset corrected coefficients (does not work well because of huge fluctuations in autocorrelation)
    # rk_offset = fit.popt[2]
    # C_raw = rk.coefficients-rk
    # range = int(6*tau_C)
    # tau_C_generalized = plots.get_T_avg(T[:range+1], C_raw[:range+1], 0)
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
recorded_system = 'V1'
setup = 'fivebins'
rec_length = '40min'
bin_size = 0.005 # sec
bin_size_ms = 5 # ms
T_0_ms = 10 # ms
bin_size_ms = int(bin_size*1000)
min_step_autocorrelation = int(T_0_ms/bin_size_ms)+1
max_step_autocorrelation = 500
min_steps_plotting = 1
max_steps_plotting = 500
neuron_list = [20, 1, 30]
# '2-338' : normal (in the new validNeurons script index 20)
# '2-303': long-range (in the new validNeurons script index 1)
# '2-357' : bursty (in the new validNeurons script index 30)

"""Load spiking data"""
# recorded_system = argv[1]
DATA_DIR = '{}/{}'.format(data_path,recorded_system)
if recorded_system == 'V1':
    DATA_DIR = '{}/neuropixels/Waksman'.format(data_path)
N_neurons_dict = {'CA1' : 28, 'retina': 111, 'culture': 48, 'V1': 142}
N_neurons = N_neurons_dict[recorded_system]
validNeurons = np.load(
    '{}/validNeurons.npy'.format(DATA_DIR)).astype(int)

for i, neuron_index in enumerate(neuron_list):
    panel = ['A','B','C'][i]
    neuron = validNeurons[neuron_index]
    print(neuron)
    spiketimes = np.load('{}/V1/spks/spiketimes-{}-{}.npy'.format(DATA_DIR,neuron[0], neuron[1]))

    # Add 5 seconds to make sure that only spikes with sufficient spiking history are considered
    t_0 = spiketimes[0] + 5.
    spiketimes = spiketimes - t_0
    spiketimes = spiketimes[spiketimes > 0]

    """Compute autocorrelation time, median ISI and CV"""
    counts_from_spikes = utl.get_binned_neuron_activity(spiketimes, bin_size)
    tau_C, fit = get_auto_correlation_time(counts_from_spikes, min_step_autocorrelation, max_step_autocorrelation, bin_size_ms)
    print(tau_C)
    rk = mre.coefficients(counts_from_spikes, dt=bin_size_ms, steps = (min_steps_plotting, max_steps_plotting))
    T = rk.steps*bin_size_ms
    rate, median_ISI, CV = get_spiking_stats(spiketimes)
    print(rate, tau_C)
    stats_dict = {'rate':rate, 'medianISI': median_ISI, 'CV' : CV, 'autocorrelation_time': tau_C}

    """Plotting"""
    rc('text', usetex=True)
    matplotlib.rcParams['font.size'] = '16.0'
    matplotlib.rcParams['xtick.labelsize'] = '16'
    matplotlib.rcParams['ytick.labelsize'] = '16'
    matplotlib.rcParams['legend.fontsize'] = '16'
    matplotlib.rcParams['axes.linewidth'] = 0.6
    # Colors
    green = sns.cubehelix_palette(8, start=.5, rot=-.75)[3]
    if i ==0:
        fig, ((ax)) = plt.subplots(nrows= 1, ncols = 1 , figsize=(3.8, 2.8))
        ax.set_ylabel(r'autocorrelation $C(T)$')
        ax.text(tau_C+5, 0.105, r'$\tau_C$')
    else:
        fig, ((ax)) = plt.subplots(nrows= 1, ncols = 1 , figsize=(3.6, 2.8))
    ax.spines['top'].set_bounds(0, 0)
    ax.spines['right'].set_bounds(0, 0)
    ax.set_xlim(0,500)
    ax.set_xticks([0,250,500])
    ax.spines['bottom'].set_bounds(0, 500)
    if neuron_index == 30:
        ax.set_xlim(0,100)
        ax.set_xticks([0,50,100])
        ax.spines['bottom'].set_bounds(0, 100)

    ax.set_xlabel(r'time lag $T$ (ms)')

    ax.step(np.append([0],T[:-1]), rk.coefficients, linewidth=1.2, color = green, where = 'post')
    ax.plot(np.append([0],T[:-1]), mre.f_exponential_offset(rk.steps, fit.tau/bin_size_ms, *fit.popt[1:]), label= 'exponential fit', color = '0.4')
    ax.axvline(x=tau_C, ymax=1.0,
                linewidth=1., linestyle='--',color = '0.4', zorder = 3)
    ax.legend(loc = ((0.2,0.62)), frameon=False)

    fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

    plt.savefig('%s/Fig8%s_autocorrelation_vs_T.pdf'%(PLOTTING_DIR, panel),
                format="pdf", bbox_inches='tight')

    plt.close()
