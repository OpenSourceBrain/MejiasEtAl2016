from __future__ import print_function, division

import os
import numpy as np
import argparse
import matplotlib.pylab as plt
import pickle

# set random set
np.random.RandomState(seed=42)

from intralaminar import intralaminar_simulation, intralaminar_analysis, intralaminar_plt
from interlaminar import interlaminar_simulation, interlaminar_activity_analysis, plot_activity_traces, \
                         calculate_interlaminar_power_spectrum, \
                         plot_interlaminar_power_spectrum
from interareal import interareal_simulation
from helper_functions import firing_rate_analysis


parser = argparse.ArgumentParser(description='Parameters for the simulation')
parser.add_argument('-noise',
                    type=float,
                    dest='noise',
                    help='Specifiy sigma of the Gausian Noise')
parser.add_argument('-analysis',
                    type=str,
                    dest='analysis',
                    help='Specifiy type of analysis to be used')
parser.add_argument('-debug',
                    dest='debug',
                    action='store_true',
                    help='Specifiy type of analysis to be used')

parser.add_argument('-nogui',
                    dest='nogui',
                    action='store_true',
                    help='No gui')
"""
Main Python file that contains the definitions for the simulation and
calls the necessary functions depending on the passed parameters.
"""
args = parser.parse_args()

# Create folder where results will be saved
if not os.path.isdir(args.analysis):
    os.mkdir(args.analysis)

# Connection between layers
wee = 1.5; wei = -3.25
wie = 3.5; wii = -2.5


# Specify membrane time constants
tau_2e = 0.006; tau_2i = 0.015
tau_5e = 0.030; tau_5i = 0.075
tau = np.array([tau_2e, tau_2i, tau_5e, tau_5i])

# sigma
sig_2e = .3; sig_2i = .3
sig_5e = .45; sig_5i = .45
sig = np.array([sig_2e, sig_2i, sig_5e, sig_5i])

if args.analysis == 'debug':
    # Call a function that plots and saves of the firing rate for the intra- and interlaminar simulation
    firing_rate_analysis()

if args.analysis == 'intralaminar':
    print('Intralaminar Simulation')
    print('-----------------------')
    # Define dt and the trial length
    dt = 2e-4
    tstop = 25 # ms
    t = np.linspace(0, tstop, tstop/dt)
    transient = 5
    # speciy number of areas that communicate with each other
    Nareas = 1

    # define interlaminar synaptic coupling strenghts
    J_2e = 0; J_2i = 0
    J_5e = 0; J_5i = 0

    J = np.array([[wee, wei, J_5e,   0],
                  [wie, wii, J_5i,   0],
                  [J_2e, 0,   wee, wei],
                  [J_2i, 0,   wie, wii]])

    # Iterate over different input strength
    Imin = 0
    Istep = 2
    Imax = 6
    # Note: the range function does not include the end
    Iexts = range(Imin, Imax + Istep, Istep)
    Ibgk = np.zeros((J.shape[0]))
    nruns = 10

    # Note: Because of the way the way intralaminar_simulation is defined only the results for L2/3
    # will be save and used for further analysis
    layer = 'L2_3'
    print('Analysing layer %s' %layer)
    # check if simulation file already exists, if not run the simulation
    simulation_file = 'intralaminar/L2_3_simulation.pckl'
    if not os.path.isfile(simulation_file):
        print('Re-calculating the simulation')
        intralaminar_simulation(args.analysis, layer, Iexts, Ibgk, nruns, t, dt, tstop,
                        J, tau, sig, args.noise, Nareas)
    else:
        print('Using the pre-saved simulation file: %s' %simulation_file)
    intralaminar_analysis(Iexts, nruns, layer, dt, transient)
    intralaminar_plt(layer)

if args.analysis == 'interlaminar_a':
    # Calculates the power spectrum for the coupled and uncoupled case for L2/3 and L5/6
    dt = 2e-4
    tstop = 600
    transient = 10

    # specify number of areas that communicate with each other
    Nareas = 1
    # Note: np.arange excludes the stop so we add dt to include the last value
    t = np.arange(0, tstop, dt)

    # define interlaminar synaptic coupling strengths
    J_2e = 1; J_2i = 0
    J_5e = 0; J_5i = 0.75

    J = np.array([[wee, wei, J_5e, 0],
                  [wie, wii, J_5i, 0],
                  [J_2e, 0, wee, wei],
                  [J_2i, 0, wie, wii]])
    Iext = np.array([[8], [0], [8], [0]])
    Ibgk = np.zeros((J.shape[0], 1))

    Nbin = 100 # pick on e very 'bin' points
    pxx_coupled_l23_bin, fxx_coupled_l23_bin, pxx_coupled_l56_bin, fxx_coupled_l56_bin = \
                        calculate_interlaminar_power_spectrum(args.analysis, t, dt, transient,
                                                              tstop, J, tau, sig, Iext, Ibgk,
                                                              args.noise, Nareas, Nbin)

    # Run simulation when the two layers are uncoupled
    # define interlaminar synaptic coupling strengths
    J_2e = 0; J_2i = 0
    J_5e = 0; J_5i = 0
    J = np.array([[wee, wei, J_5e, 0],
                  [wie, wii, J_5i, 0],
                  [J_2e, 0, wee, wei],
                  [J_2i, 0, wie, wii]])

    pxx_uncoupled_l23_bin, fxx_uncoupled_l23_bin, pxx_uncoupled_l56_bin, fxx_uncoupled_l56_bin = \
        calculate_interlaminar_power_spectrum(args.analysis, t, dt, transient,
                                           tstop, J, tau, sig, Iext, Ibgk,
                                           args.noise, Nareas, Nbin)
    # Plot spectrogram
    plot_interlaminar_power_spectrum(fxx_uncoupled_l23_bin, fxx_coupled_l23_bin,
                                  pxx_uncoupled_l23_bin, pxx_coupled_l23_bin,
                                  fxx_uncoupled_l56_bin, fxx_coupled_l56_bin,
                                  pxx_uncoupled_l56_bin, pxx_coupled_l56_bin,
                                  args.analysis)


if args.analysis == 'interlaminar_b':
    # Calculates the spectogram and 30 traces of actvity in layer 5/6
    # Define dt and the trial length
    dt = 2e-4
    tstop = 6000
    transient = 10
    # specify number of areas that communicate with each other
    Nareas = 1
    # Note: np.arange excludes the stop so we add dt to include the last value
    t = np.arange(dt+transient, tstop + dt, dt)

    # define interlaminar synaptic coupling strengths
    J_2e = 1; J_2i = 0
    J_5e = 0; J_5i = 0.75

    J = np.array([[wee, wei, J_5e,   0],
                  [wie, wii, J_5i,   0],
                  [J_2e, 0,   wee, wei],
                  [J_2i, 0,   wie, wii]])

    Iext = np.array([[6], [0], [8], [0]])
    Ibgk = np.zeros((J.shape[0], 1))

    # frequencies of interest
    min_freq5 = 4 # alpha range
    min_freq2 = 30 # gama range

    # check if file with simulation exists, if not calculate the simulation
    if not os.path.isfile(os.path.join(args.analysis, 'simulation.pckl')):
        rate = interlaminar_simulation(args.analysis, t, dt, tstop, J, tau, sig, Iext, Ibgk, args.noise, Nareas)
    else:
        # load pickle file with results
        picklename = os.path.join(args.analysis, 'simulation.pckl')
        with open(picklename, 'rb') as filename:
            rate = pickle.load(filename)

    # Analyse and Plot traces of activity in layer 5/6
    segment5, segment2, segindex, numberofzones = interlaminar_activity_analysis(rate, transient, dt, t, min_freq5)
    plot_activity_traces(dt, segment5, segindex, args.analysis)

    # Analyse and Plot spectrogram of layer L2/3
    # For now, ignore this function as I cannot generate the correct output
    # ff, tt, Sxx = interlaminar_analysis_periodeogram(rate, segment2, transient, dt, min_freq2, numberofzones)
    # plot_spectrogram(ff, tt, Sxx)


if args.analysis == 'interareal':
    dt = 2e-04
    tstop = 40
    transient = 5
    # speciy number of areas that communicate with each other
    Nareas = 2
    t = np.arange(dt, tstop + dt - transient, dt)

    # define interlaminar synaptic coupling strenghts
    J_2e = 1; J_2i = 0
    J_5e = 0; J_5i = 0.75

    J = np.array([[wee, wei, J_5e,   0],
                  [wie, wii, J_5i,   0],
                  [J_2e, 0,   wee, wei],
                  [J_2i, 0,   wie, wii]])


    # Interareal connectivity

    W = np.zeros((2, 2, 4, 2))
    # W(a, b, c, d), where
    # a = post. area
    # b = pres. area
    # c = post. layer
    # d = pres. layer

    W[1, 0, 0, 0] = 1       # V1 to V4, supra to supra
    W[1, 0, 1, 0] = 0       # V1 to V4, supra to infra
    W[0, 1, 0, 1] = s       # V4 to V1, infra to supra excit
    W[0, 1, 2, 1] = .5      # other option, same result ?
    W[0, 1, 1, 1] = (1 - s) # V4 to V1, infra to infra excit
    W[0, 1, 3, 2] = .5      # other option same result

    # background and injected current
    Ibgk = 2 * np.array([[1, 1], [0, 0], [2, 2], [0, 0]])
    Iext = 15 * np.array([[1, 0], [0, 0], [1, 0], [0, 0]])

    # at rest
    interareal_simulation(t, dt, tstop, J, W, tau, Iext, Ibkg, sig, args.noise)
    # microstimulation


if not args.nogui:
    plt.show()

