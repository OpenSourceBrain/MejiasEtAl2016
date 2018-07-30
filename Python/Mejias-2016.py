from __future__ import print_function, division

import os
import numpy as np
import argparse
import matplotlib.pylab as plt

# set random set
np.random.RandomState(seed=42)

from intralaminar import intralaminar_simulation, intralaminar_analysis, intralaminar_plt
from helper_scripts import debug_neuroml

parser = argparse.ArgumentParser(description='Parameters for the simulation')
parser.add_argument('-noise',
                    type=float,
                    dest='noise',
                    help='Specifiy sigma of the Gausian Noise')
parser.add_argument('-analysis',
                    type=str,
                    dest='analysis',
                    help='Specifiy type of analysis to be used')

parser.add_argument('-nogui',
                    dest='nogui',
                    action='store_true',
                    help='No gui')

args = parser.parse_args()

# Create folder where results will be saved
if not os.path.isdir(args.analysis):
    os.mkdir(args.analysis)

# Connection between layers
wee = 1.5; wei = -3.25
wie = 3.5; wii = -2.5

J_2e = 0; J_2i = 0
J_5e = 0; J_5i = 0

J = np.array([[wee, wei, J_5e,   0],
              [wie, wii, J_5i,   0],
              [J_2e, 0,   wee, wei],
              [J_2i, 0,   wie, wii]])

# Specify membrane time constants
tau_2e = 0.006; tau_2i = 0.015
tau_5e = 0.030; tau_5i = 0.075
tau = np.array([[tau_2e], [tau_2i], [tau_5e], [tau_5i]])

# sigma
sig_2e = .3; sig_2i = .3
sig_5e = .45; sig_5i = .45
sig = np.array([[sig_2e], [sig_2i], [sig_5e], [sig_5i]])

if args.analysis == 'debug_neuroML':
    dt = 2e-4
    tstop = 1 # ms
    t = np.linspace(0, tstop, tstop/dt)
    Iexts = [0]
    nruns = [1]
    # For testing purpose test only L2_3 layer
    layer = 'L2_3'

    debug_neuroml(args.analysis, layer, t, dt, tstop, J,
                  tau, sig, Iexts, nruns, args.noise, args.nogui)

if args.analysis == 'intralaminar':
    # Define dt and the trial length
    dt = 2e-4
    tstop = 25 # ms
    t = np.linspace(0, tstop, tstop/dt)
    transient = 5

    # Iterate over different input strength
    Imin = 0
    Istep = 2
    Imax = 6
    # Note: the range function does not include the end
    Iexts = range(Imin, Imax + Istep, Istep)
    nruns = 10

    # Note: Because of the way the way intralaminar_simulation is defined only the results for L2/3
    # will be save and used for further analysis
    layer = 'L2_3'
    # check if simulation file already exists, if not run the simulation
    intralaminar_simulation(args.analysis, layer, Iexts, nruns, t, dt, tstop,
                        J, tau, sig, args.noise)
    intralaminar_analysis(Iexts, nruns, layer, dt, transient)
    intralaminar_plt(layer)


if not args.nogui:
    plt.show()

