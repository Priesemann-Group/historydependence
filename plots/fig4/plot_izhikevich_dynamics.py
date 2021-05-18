"""
Simulation of an Izhikevich neuron.

Run as:

$ python plot_izhikevich_dynamics.py nest --plot-figure

"""

import numpy as np
from numpy import arange
from pyNN.utility import get_simulator, init_logging, normalized_filename
from pyNN.random import NumpyRNG
import matplotlib.pyplot as plt
from os.path import isfile, isdir, realpath, dirname, exists
import os
from sys import exit, stderr, argv, path, modules

# === Configure the simulator ================================================
noise_std = 0.001
sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file.", {"action": "store_true"}),
                             ("--debug", "Print debugging information"))

if options.debug:
    init_logging(None, debug=True)

sim.setup(timestep=0.01, min_delay=1.0)
Trec_ms = 200. # simulation time in ms

# === Build and instrument the network =======================================

neurons = sim.Population(1, sim.Izhikevich(a=0.02, b=0.2, c=-50, d=2, i_offset=[0.0]))

noise = sim.NoisyCurrentSource(mean=0.011, stdev=noise_std, start=1.0, stop=Trec_ms, dt=1.0)

noise.inject_into(neurons)

neurons.record(['v','u'])

neurons.initialize(v=-70.0, u=-14.0)

# === Run the simulation =====================================================
sim.run(Trec_ms)

# === Save the results, optionally plot a figure =============================
filename = normalized_filename("Results", "Izhikevich", "pkl",
                               options.simulator, sim.num_processes())

if options.plot_figure:
    from pyNN.utility.plotting import Figure, Panel
    figure_filename = filename.replace("pkl", "pdf")
    data = neurons.get_data().segments[0]
    v = data.filter(name="v")[0]
    u = data.filter(name="u")[0]
    Figure(
        Panel(v, ylabel="Membrane potential (mV)", xticks=True,
              xlabel="Time (ms)", yticks=True),
        Panel(u, ylabel="Recovery variable (mV)", xticks=True,
              xlabel="Time (ms)", yticks=True),
        #Panel(u, ylabel="u variable (units?)"),
        annotations="Simulated with %s" % options.simulator.upper()
    ).save(figure_filename)
    print(figure_filename)
