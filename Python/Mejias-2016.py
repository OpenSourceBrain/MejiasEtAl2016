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
parser.add_argument('-tau_e',
                    type=float,
                    dest='tau_e',
                    help='Excitatory membrane time constant (tau_e)')
parser.add_argument('-tau_i',
                    type=float,
                    dest='tau_i',
                    help='Inhibitory membrane time constant (tau_i)')
parser.add_argument('-sei',
                    type=float,
                    dest='sei',
                    help='Deviation for the Gaussian white noise (s_ei)')
parser.add_argument('-layer',
                    type=str,
                    dest='layer',
                    help='Layer of interest')
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
wee = 1.5
wei = -3.25
wie = 3.5
wii = -2.5

if args.analysis == 'debug_neuroML':
    dt = 2e-4
    tstop = 1 # ms
    t = np.linspace(0, tstop, tstop/dt)
    Iexts = [0]
    nruns = [1]

    debug_neuroml(args.analysis, args.layer, t, dt, tstop, wee, wei, wie, wii,
                  args.tau_e, args.tau_i, args.sei, Iexts, nruns, args.noise, args.nogui)

if args.analysis == 'intralaminar':
    dt = 2e-4
    tstop = 25 # ms
    t = np.linspace(0, tstop, tstop/dt)

    # Iterate over different input strength
    Imin = 0
    Istep = 2
    Imax = 6
    # Note: the range function does not include the end
    Iexts = range(Imin, Imax + Istep, Istep)
    nruns = 10
    intralaminar_simulation(args.analysis, args.layer, Iexts, nruns, t, dt, tstop, wee, wei, wie, wii,
                            args.tau_e, args.tau_i, args.sei, args.noise)
    intralaminar_analysis(Iexts, nruns, args.layer, dt, args.nogui)
    intralaminar_plt(args.layer, args.nogui)


if not args.nogui:
    plt.show()

