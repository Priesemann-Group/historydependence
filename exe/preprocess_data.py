from sys import stderr, exit, argv
import numpy as np
from scipy.io import loadmat
import os
from os.path import isfile, isdir, realpath, dirname, exists


def preprocessStringerNeuropixelsData(data_path, output_path):
    # Only accept neurons with at least 40 minutes recording length
    minRecLength = 2400.
    rawDATA_DIR = '{}/neuropixels/raw'.format(data_path)

    # rawDATA_DIR = '/data.nst/share/data/stringer_spikes_neuropixels'
    basenameDataFile = 'spks{}_Feb18.mat'
    probeLocationsFileName = 'probeLocations.mat'
    probeBordersFileName = 'probeBorders.mat'
    numberOfProbes = 8
    probeLocations = loadmat('{}/{}'.format(rawDATA_DIR,
                                            probeLocationsFileName))

    probeBorders = loadmat('{}/{}'.format(rawDATA_DIR,
                                          probeBordersFileName), squeeze_me=True)
    # mouseNumber = 1 is Waksman
    mouseNumber = 1
    mouseName = str(
        probeLocations['probeLocations'][0][mouseNumber]['mouseName'][0])
    saveDATA_DIR = '{}/neuropixels/{}/V1'.format(output_path, mouseName)
    if not isdir('{}/neuropixels'.format(output_path)):
        os.mkdir('{}/neuropixels'.format(output_path))
    if not isdir(saveDATA_DIR):
        os.mkdir(saveDATA_DIR)
    if not isdir('{}/spks'.format(saveDATA_DIR)):
        os.mkdir('{}/spks'.format(saveDATA_DIR))

    # print("##### Mouse: {}".format(mouseName))
    spks = loadmat('{}/spks/{}'.format(rawDATA_DIR,
                                       basenameDataFile.format(mouseName)), squeeze_me=True)
    # find detailed areas from which was recorded in the respective mouse (Waksman)
    detailedAreas = np.array([])
    for probeNumber in range(numberOfProbes):
        ccfOntology = [name[0][0] for name in probeLocations['probeLocations']
                       [0][mouseNumber]['probe'][0][probeNumber]['ccfOntology']]
        detailedAreas = np.append(detailedAreas, np.unique(ccfOntology))
    detailedAreas = np.unique(detailedAreas)
    detailedAreasPrimaryVisualCortex = ['VISp2/3', 'VISp4', 'VISp5', 'VISp6b', 'VISp6a']

    # Save only spiketimes of neurons in primary visual cortex, and that satisfy the rate requirements
    validNeurons = []
    for probeNumber in range(numberOfProbes):
        print("### Probe {}".format(probeNumber))
        ccfCoords = probeLocations['probeLocations'][0][mouseNumber]['probe'][0][probeNumber]['ccfCoords']
        ccfOntology = [name[0][0] for name in probeLocations['probeLocations']
                       [0][mouseNumber]['probe'][0][probeNumber]['ccfOntology']]
        unsortedSptimes = spks['spks'][probeNumber][0]
        clusterIdentities = np.array(spks['spks'][probeNumber][1]) - 1  # start at 0 instead of 1
        # cluster heights in microns
        wHeights = spks['spks'][probeNumber][2]
        # Load spikes
        sptimes = [[] for cli in np.unique(clusterIdentities)]
        for sptime, cli in zip(unsortedSptimes, clusterIdentities):
            sptimes[cli] += [float(sptime)]
        Nneurons = len(wHeights)
        for neuron in range(Nneurons):
            # Spacing of electrodes is 20 mm, but two electrodes have the same height
            ccfIndex = int(wHeights[neuron] / 20 * 2)
            detailedArea = ccfOntology[ccfIndex]
            if detailedArea in detailedAreasPrimaryVisualCortex:
                spiketimes_neuron = np.sort(sptimes[neuron])
                t_start = spiketimes_neuron[0]
                t_end = spiketimes_neuron[-1]
                Trec = t_end - t_start
                rate = len(spiketimes_neuron) / Trec
                if (rate < maxRate and Trec > minRecLength) and rate > minRate:
                    validNeurons += [[probeNumber, neuron]]
                    np.save('{}/spks/spiketimes-{}-{}.npy'.format(saveDATA_DIR,
                                                          probeNumber, neuron), np.sort(spiketimes_neuron))
    # Save dictionary of valid neurons used for the analysis
    np.save('{}/validNeurons.npy'.format(saveDATA_DIR), validNeurons)


def preprocessRetinaData(data_path, output_path):
    sampling_rate = 10000.  # 10 kHz
    rawDATA_DIR = '{}/retina/raw/mode_paper_data/unique_natural_movie'.format(data_path)
    saveDATA_DIR = '{}/retina'.format(output_path)
    if not isdir(saveDATA_DIR):
        os.mkdir(saveDATA_DIR)
    if not isdir('{}/spks'.format(saveDATA_DIR)):
        os.mkdir('{}/spks'.format(saveDATA_DIR))
    data = loadmat('{}/data.mat'.format(rawDATA_DIR))

    # Neuron list
    neurons = data['data'][0][0][2][0][0][2][0]
    N_neurons = neurons[-1]

    # find valid neurons with 0.5Hz < rate < 10 Hz and save their spiketimes
    validNeurons = []
    for neuron in range(N_neurons):
        spiketimes_neuron = data['data'][0][0][2][0][0][1][0][neuron][0] / sampling_rate
        np.save('{}/spks/spiketimes_neuron{}.npy'.format(saveDATA_DIR, neuron), spiketimes_neuron)
        t_start = spiketimes_neuron[0]
        t_end = spiketimes_neuron[-1]
        Trec = t_end - t_start
        rate = spiketimes_neuron.size/Trec
        if rate > minRate and rate < maxRate:
            validNeurons += [neuron]
    np.save('{}/validNeurons.npy'.format(saveDATA_DIR), validNeurons)

    # Start and end times movie (probably)
    # T_0 = data['data'][0][0][3][0][0][0][0][0] / sampling_rate
    # T_f = data['data'][0][0][3][0][0][1][1][0] / sampling_rate
    # T = T_f - T_0

    # Description
    # data['data'][0][0][0]

    # Date
    # data['data'][0][0][1]

    # Sampling rate
    # print(data['data'][0][0][2][0][0][0])

    # Full data
    # data['data'][0][0][2][0][0][1]

    # Short data, but not really sure what this is. Spiketimes are not the same
    # data['data'][0][0][2][0][0][3]


def preprocessCA1Data(data_path, output_path):
    rawDATA_DIR = '{}/CA1/raw'.format(data_path)
    saveDATA_DIR = '{}/CA1'.format(output_path)
    if not isdir(saveDATA_DIR):
        os.mkdir(saveDATA_DIR)
    if not isdir('{}/spks'.format(saveDATA_DIR)):
        os.mkdir('{}/spks'.format(saveDATA_DIR))
    data = loadmat('{}/ec014.277.spike_ch.mat'.format(rawDATA_DIR))
    sample_rate = 20000.  # 20 kHz sampling rate in seconds
    sptimes = data['sptimes'][0] / sample_rate
    singleunit = data['singleunit'][0]
    end_times = data['t_end'].flatten() / sample_rate
    Nneurons = 85
    validNeurons = []
    for neuron in range(Nneurons):
        if singleunit[neuron] == 1:
            spiketimes_neuron = sptimes[neuron].flatten()
            np.save('{}/spks/spiketimes_neuron{}.npy'.format(saveDATA_DIR, neuron), spiketimes_neuron)
            t_start = spiketimes_neuron[0]
            t_end = end_times[neuron]
            Trec = t_end - t_start
            rate = spiketimes_neuron.size/Trec
            if rate > minRate and rate < maxRate:
                validNeurons += [neuron]
    np.save('{}/validNeurons.npy'.format(saveDATA_DIR), validNeurons)


def preprocessCultureData(data_path, output_path):
    rawDATA_DIR = '{}/culture/raw'.format(data_path)
    saveDATA_DIR = '{}/culture'.format(output_path)
    if not isdir(saveDATA_DIR):
        os.mkdir(saveDATA_DIR)
    if not isdir('{}/spks'.format(saveDATA_DIR)):
        os.mkdir('{}/spks'.format(saveDATA_DIR))
    spiketimes1 = np.loadtxt("{}/L_Prg035_txt_nounstim.txt".format(rawDATA_DIR))
    spiketimes2 = np.loadtxt("{}/L_Prg036_txt_nounstim.txt".format(rawDATA_DIR))
    spiketimes3 = np.loadtxt("{}/L_Prg037_txt_nounstim.txt".format(rawDATA_DIR))
    spiketimes4 = np.loadtxt("{}/L_Prg038_txt_nounstim.txt".format(rawDATA_DIR))
    spiketimes5 = np.loadtxt("{}/L_Prg039_txt_nounstim.txt".format(rawDATA_DIR))

    spiketimes = np.append(spiketimes1, spiketimes2, axis=0)
    spiketimes = np.append(spiketimes, spiketimes3, axis=0)
    spiketimes = np.append(spiketimes, spiketimes4, axis=0)
    spiketimes = np.append(spiketimes, spiketimes5, axis=0)

    sample_rate = 24.03846169
    times = spiketimes.transpose()[0]
    neurons = spiketimes.transpose()[1]
    # spiketimes in seconds
    times = times/sample_rate/1000
    validNeurons = []
    for neuron in np.arange(1, 61):
        spiketimes_neuron = times[np.where(neurons == neuron)[0]]
        np.save('{}/spks/spiketimes_neuron{}.npy'.format(saveDATA_DIR, neuron), spiketimes_neuron)
        t_start = spiketimes_neuron[0]
        t_end = spiketimes_neuron[-1]
        Trec = t_end - t_start
        rate = spiketimes_neuron.size/Trec
        if (rate < maxRate and rate > minRate):
            validNeurons += [neuron]
    np.save('{}/validNeurons.npy'.format(saveDATA_DIR), validNeurons)
    len(validNeurons)


# During preprocessing, only neurons with an average firing rate between minRate and maxRate (in Hz) are considered for the analysis.
minRate = 0.5
maxRate = 10.
recorded_system = argv[1]
# If data_path not specified, use analysis_data of the repository
if len(argv) > 2:
    data_path = argv[2]
else:
    CODE_DIR = '{}/..'.format(dirname(realpath(__file__)))
    data_path = '{}/data'.format(CODE_DIR)
# If output_path not specified, use analysis_data of the repository
if len(argv) > 3:
    output_path = argv[3]
else:
    CODE_DIR = '{}/..'.format(dirname(realpath(__file__)))
    output_path = '{}/data'.format(CODE_DIR)

if __name__ == "__main__":
    if recorded_system == 'V1':
        preprocessStringerNeuropixelsData(data_path, output_path)
    if recorded_system == 'retina':
        preprocessRetinaData(data_path, output_path)
    if recorded_system == 'CA1':
        preprocessCA1Data(data_path, output_path)
    if recorded_system == 'culture':
        preprocessCultureData(data_path, output_path)
