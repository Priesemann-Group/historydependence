"""Functions"""
from scipy.io import loadmat
import seaborn.apionly as sns
from scipy.optimize import bisect
from matplotlib import rc
import matplotlib.lines as mlines
import pylab as plt
import matplotlib
import numpy as np
from sys import exit, stderr, argv, path, modules
from os.path import isfile, isdir, realpath, dirname, exists

PLOTTING_DIR = dirname(realpath(__file__))
CODE_DIR = '{}/../..'.format(PLOTTING_DIR)
recorded_system = argv[1]
if len(argv) > 2:
    data_path = argv[2]
else:
    data_path = '{}/data'.format(CODE_DIR)

##### Plot params #####

# Colors
main_red = sns.color_palette("RdBu_r", 15)[12]
main_blue = sns.color_palette("RdBu_r", 15)[1]
soft_red = sns.color_palette("RdBu_r", 15)[10]
soft_blue = sns.color_palette("RdBu_r", 15)[4]

rc('text', usetex=True)
matplotlib.rcParams['font.size'] = '13.0'
matplotlib.rcParams['xtick.labelsize'] = '13.0'
matplotlib.rcParams['ytick.labelsize'] = '13.0'
matplotlib.rcParams['legend.fontsize'] = '13.0'
matplotlib.rcParams['axes.linewidth'] = 0.6

fig, ax = plt.subplots(1, 1, figsize=(3., 2.8))

##### Unset Borders #####

for side in ['right', 'top', 'left']:
    ax.spines[side].set_visible(False)

##### remove axis ticks #####

ax.xaxis.set_ticks_position('none')  # tick markers
ax.yaxis.set_ticks_position('none')

##### unset ticks and labels #####
plt.xticks([])  # labels
plt.yticks([])
plt.setp([a.get_xticklabels() for a in [ax]], visible=False)
plt.setp([a.get_yticklabels() for a in [ax]], visible=False)

"""Load spike data"""

"""For Retina data"""
if recorded_system == 'retina':
    data_path_spikes = "{}/retina/spks".format(data_path)

    ax.set_title("salamander retina", fontsize=17)

    ##### x-axis ####
    ax.set_xlim((52, 62))
    ax.spines['bottom'].set_bounds(60, 62)
    ax.text(60.7, -14, '2s')

    ##### y-axis ####
    ax.set_ylabel(r'neuron \#')
    ax.set_ylim((-1.5, 152.5))
    ax.spines['left'].set_bounds(1, 152)

    ##### plot spikes ####
    for neuron in range(152):
        spiketimes = np.load(
            '%s/spiketimes_neuron%d.npy' % (data_path_spikes, neuron))
        ax.scatter(spiketimes[spiketimes < 63.], np.zeros(len(spiketimes[spiketimes < 63.])) + neuron + 1, s=0.2, color='0', zorder=1)

"""For hippocampus data"""
if recorded_system == 'CA1':
    data_path_spikes = "{}/CA1/spks".format(data_path)

    ax.set_title("rat dorsal hippocampus (CA1)", fontsize=17)

    ##### x-axis ####
    ax.set_xlim((52, 62))
    ax.spines['bottom'].set_bounds(60, 62)
    ax.text(60.7, -8, '2s')

    ##### y-axis ####
    ax.set_ylabel(r'neuron \#')
    ax.set_ylim((0.5, 85.5))
    ax.spines['left'].set_bounds(1, 85)

    # Check if neuron is stored (single units only)(isfile), then it is also plotted
    for neuron in range(85):
        if isfile('%s/spiketimes_neuron%d.npy'%(data_path_spikes, neuron)):
            spiketimes = np.load(
                '%s/spiketimes_neuron%d.npy' % (data_path_spikes, neuron))
            ax.scatter(spiketimes[spiketimes < 63.], np.zeros(len(spiketimes[spiketimes < 63.])) + neuron + 1, s=0.2, color='0', zorder=1)


"""For in culture data"""
if recorded_system == 'culture':
    data_path_spikes = "{}/culture/spks".format(data_path)

    ax.set_title("rat cortical culture", fontsize=17)

    ##### x-axis ####
    ax.set_xlim((72, 82))
    ax.spines['bottom'].set_bounds(80, 82)
    ax.text(80.7, -7, '2s')

    ##### y-axis ####
    ax.set_ylabel(r'neuron \#')
    ax.set_ylim((-1.5, 60.5))
    ax.spines['left'].set_bounds(1, 60)

    ##### plot spikes ####
    for neuron in np.arange(1, 61):
        spiketimes = np.load('%s/spiketimes_neuron%d.npy'%(data_path_spikes, neuron))
        ax.scatter(spiketimes[spiketimes < 83.], np.zeros(
            len(spiketimes[spiketimes < 83.])) + neuron, s=0.2, color='0', zorder=1)

if recorded_system == 'V1':
    data_path_spikes = "{}/neuropixels/Waksman/V1/spks".format(data_path)
    validNeurons = np.load("{}/neuropixels/Waksman/V1/validNeurons.npy".format(data_path))

    ax.set_title("mouse primary visual cortex", fontsize=17)

    ##### x-axis ####
    ax.set_xlim((352, 362))
    ax.spines['bottom'].set_bounds(360, 362)
    ax.text(360.7, -13, '2s')

    ##### y-axis ####
    ax.set_ylabel(r'neuron \#')
    ax.set_ylim((-1.5, 142.5))
    ax.spines['left'].set_bounds(1, 142)

    ##### plot spikes of analyzed neurons only ####
    neuron_ID = 1
    for neuron in validNeurons:
        spiketimes = np.load('{}/spiketimes-{}-{}.npy'.format(data_path_spikes,
                                                               neuron[0], neuron[1]))
        if neuron_ID == 1:
            T_0 = spiketimes[0]
        spiketimes = spiketimes - T_0
        ax.scatter(spiketimes[spiketimes < 363.], np.zeros(len(spiketimes[spiketimes < 363.])) +
                   neuron_ID, s=0.2, color='0', zorder=1)
        neuron_ID += 1

fig.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

plt.savefig('{}/Fig7_spiking_activity_{}.png'.format(PLOTTING_DIR, recorded_system),
            format="png", dpi=400, bbox_inches='tight')

plt.show()
plt.close()
