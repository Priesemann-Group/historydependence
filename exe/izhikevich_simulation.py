"""
A simulation of an Izhikevich neurons.

Run as:

$ python izhikevich_simulation.py nest

"""

import numpy as np
from numpy import arange
from pyNN.utility import get_simulator, init_logging, normalized_filename
from pyNN.random import NumpyRNG
import matplotlib.pyplot as plt
from os.path import isfile, isdir, realpath, dirname, exists
import os
from sys import exit, stderr, argv, path, modules

CODE_DIR = '{}/..'.format(dirname(realpath(__file__)))
path.insert(1, '{}/src'.format(CODE_DIR))

# === Configure the simulator ================================================
noise_std = 0.001
sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file.", {"action": "store_true"}),
                             ("--debug", "Print debugging information"))

if options.debug:
    init_logging(None, debug=True)

sim.setup(timestep=0.01, min_delay=1.0)
Trec_min = 20 # in minutes
Trec_s = Trec_min * 60 # in seconds
Trec_ms = Trec_s * 1000. # simulation time in ms

# === Build and instrument the network =======================================

neurons = sim.Population(1, sim.Izhikevich(a=0.02, b=0.2, c=-50, d=2, i_offset=[0.0]))

noise = sim.NoisyCurrentSource(mean=0.011, stdev=noise_std, start=1.0, stop=Trec_ms, dt=1.0)

noise.inject_into(neurons)

neurons.record(['spikes'])

neurons.initialize(v=-70.0, u=-14.0)

# === Run the simulation =====================================================
sim.run(Trec_ms)
data_out = neurons.get_data()
spiketrain = data_out.segments[0].spiketrains
spiketimes = np.array(spiketrain[0])/1000. #transform to seconds
t_0 = 0.1 # neglect the first 100 milliseconds
spiketimes = spiketimes - t_0
spiketimes = spiketimes[spiketimes > 0]
np.save('%s/data/izhikevich_neuron/spiketimes_%dmin.npy'%(CODE_DIR,  Trec_min), spiketimes)
sim.end()
