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
from helper_functions import firing_rate_analysis, get_network_configuration


"""
Main Python file that contains the definitions for the simulation and
calls the necessary functions depending on the passed parameters.
"""


def getArguments():
    parser = argparse.ArgumentParser(description='Parameters for the simulation')
    parser.add_argument('-sigmaoverride',
                        type=float,
                        dest='sigmaoverride',
                        default=None,
                        help='Override sigma of the Gaussian noise for ALL populations (if None leave them as is)')
    parser.add_argument('-analysis',
                        type=str,
                        dest='analysis',
                        default='debug',
                        help='Specify type of analysis to be used')
    parser.add_argument('-debug',
                        dest='debug',
                        action='store_true',
                        help='Specify whether to generate simulations for debugging')
    parser.add_argument('-noconns',
                        dest='noconns',
                        action='store_true',
                        help='Specify whether to remove connections (DEBUG MODE ONLY!)')
    parser.add_argument('-testduration',
                        type=float,
                        dest='testduration',
                        default=1000.,
                        help='Duration of test simulation (DEBUG MODE ONLY!)')
    parser.add_argument('-dt',
                        type=float,
                        dest='dt',
                        default=2e-4,
                        help='Timestep (dt) of simulation')
    parser.add_argument('-initialrate',
                        type=float,
                        dest='initialrate',
                        default=-1,
                        help='Initial rate of test simulation, if negative, use a random value (default) (DEBUG MODE ONLY!)')
    parser.add_argument('-nogui',
                        dest='nogui',
                        action='store_true',
                        help='No gui')
    return parser.parse_args()

if __name__ == "__main__":
    args = getArguments()

    # Create folder where results will be saved
    if not os.path.isdir(args.analysis):
        os.mkdir(args.analysis)

    if args.analysis == 'debug':
        print('-----------------------')
        print('Debugging')
        print('-----------------------')
        # Call a function that plots and saves of the firing rate for the intra- and interlaminar simulation
        print('Running debug simulation/analysis with %s'%args)
        dt = args.dt
        firing_rate_analysis(args.noconns, args.testduration, args.sigmaoverride, args.initialrate, dt)

    if args.analysis == 'intralaminar':
        print('-----------------------')
        print('Intralaminar Analysis')
        print('-----------------------')
        # Define dt and the trial length
        dt = args.dt
        tstop = 25 # s
        t = np.linspace(0, tstop, tstop/dt)
        transient = 5
        # speciy number of areas that communicate with each other
        Nareas = 1

        tau, sig, J, Iext, Ibgk = get_network_configuration('intralaminar', noconns=False)
        nruns = 10

        # Note: Because of the way the way intralaminar_simulation is defined only the results for L2/3
        # will be save and used for further analysis
        layer = 'L23'
        print('    Analysing layer %s' %layer)
        # check if simulation file already exists, if not run the simulation
        simulation_file = 'intralaminar/L23_simulation.pckl'
        if not os.path.isfile(simulation_file):
            print('    Re-calculating the simulation')
            simulation = intralaminar_simulation(args.analysis, layer, Iext, Ibgk, nruns, t, dt, tstop,
                            J, tau, sig, args.sigmaoverride, Nareas)
        else:
            print('    Loading the pre-saved simulation file: %s' %simulation_file)
            picklename = os.path.join('intralaminar', layer + '_simulation.pckl')
            with open(picklename, 'rb') as file1:
                simulation = pickle.load(file1)

        psd_analysis = intralaminar_analysis(simulation, Iext, nruns, layer, dt, transient)
        intralaminar_plt(psd_analysis)

    if args.analysis == 'interlaminar_a':
        print('-----------------------')
        print('Interlaminar Analysis')
        print('-----------------------')
        # Calculates the power spectrum for the coupled and uncoupled case for L2/3 and L5/6
        dt = args.dt
        tstop = 600
        transient = 10

        # specify number of areas that communicate with each other
        Nareas = 1
        # Note: np.arange excludes the stop so we add dt to include the last value
        t = np.arange(0, tstop, dt)

        tau, sig, J_conn, Iext_conn, Ibgk_conn = get_network_configuration('interlaminar_a', noconns=False)
        Nbin = 100 # pick one very 'bin' points

        # Calculate the rate
        rate_conn = interlaminar_simulation(args.analysis, t, dt, tstop, J_conn, tau, sig, Iext_conn, Ibgk_conn, args.sigmaoverride, Nareas)
        pxx_coupled_l23_bin, fxx_coupled_l23_bin, pxx_coupled_l56_bin, fxx_coupled_l56_bin = \
                            calculate_interlaminar_power_spectrum(rate_conn, dt, transient, Nbin)

        # Run simulation when the two layers are uncoupled
        tau, sig, J_noconn, Iext_noconn, Ibgk_noconn = get_network_configuration('interlaminar_u', noconns=False)

        rate_noconn = interlaminar_simulation(args.analysis, t, dt, tstop, J_noconn, tau, sig, Iext_noconn, Ibgk_conn, args.sigmaoverride, Nareas)
        pxx_uncoupled_l23_bin, fxx_uncoupled_l23_bin, pxx_uncoupled_l56_bin, fxx_uncoupled_l56_bin = \
                           calculate_interlaminar_power_spectrum(rate_noconn, dt, transient, Nbin)
        # Plot spectrogram
        plot_interlaminar_power_spectrum(fxx_uncoupled_l23_bin, fxx_coupled_l23_bin,
                                      pxx_uncoupled_l23_bin, pxx_coupled_l23_bin,
                                      fxx_uncoupled_l56_bin, fxx_coupled_l56_bin,
                                      pxx_uncoupled_l56_bin, pxx_coupled_l56_bin,
                                      args.analysis)

        # Pickle the results rate over time
        # Transform the results so that they are saved in a dic (similar to NeuroML output)
        pyrate = {'L23_E_Py/conn': rate_conn[0, :, 0],
               'L23_I_Py/conn': rate_conn[1, :, 0],
               'L56_E_Py/conn': rate_conn[2, :, 0],
               'L56_I_Py/conn': rate_conn[3, :, 0],
               'L23_E_Py/unconn': rate_noconn[0, :, 0],
               'L23_I_Py/unconn': rate_noconn[1, :, 0],
               'L56_E_Py/unconn': rate_noconn[2, :, 0],
               'L56_I_Py/unconn': rate_noconn[3, :, 0],
               'ts': t
               }
        picklename = os.path.join('debug', args.analysis, 'simulation.pckl')
        if not os.path.exists(picklename):
            os.mkdir(os.path.dirname(picklename))
        with open(picklename, 'wb') as filename:
            pickle.dump(pyrate, filename)


        print('    Done Analysis!')


    if args.analysis == 'interlaminar_b':
        print('-----------------------')
        print('Interlaminar Simulation')
        print('-----------------------')
        # Calculates the spectogram and 30 traces of actvity in layer 5/6
        # Define dt and the trial length
        dt = args.dt
        tstop = 6000
        transient = 10
        # specify number of areas that communicate with each other
        Nareas = 1
        # Note: np.arange excludes the stop so we add dt to include the last value
        t = np.arange(dt+transient, tstop + dt, dt)

        tau, sig, J, Iext, Ibgk = get_network_configuration('interlaminar_b', noconns=False)
        # frequencies of interest
        min_freq5 = 4 # alpha range
        min_freq2 = 30 # gama range

        # check if file with simulation exists, if not calculate the simulation
        simulation_file = os.path.join(args.analysis, 'simulation.pckl')
        if not os.path.isfile(simulation_file):
            print('    Re-calculating the simulation')
            rate = interlaminar_simulation(args.analysis, t, dt, tstop, J, tau, sig, Iext, Ibgk, args.sigmaoverride, Nareas)
        else:
            print('    Loading the pre-saved simulation file: %s' %simulation_file)
            with open(simulation_file, 'rb') as filename:
                rate = pickle.load(filename)

        # Analyse and Plot traces of activity in layer 5/6
        segment5, segment2, segindex, numberofzones = interlaminar_activity_analysis(rate, transient, dt, t, min_freq5)
        plot_activity_traces(dt, segment5, segindex, args.analysis)

        # Analyse and Plot spectrogram of layer L2/3
        # For now, ignore this function as I cannot generate the correct output
        from interlaminar import interlaminar_analysis_periodeogram
        ff, tt, Sxx = interlaminar_analysis_periodeogram(rate, segment2, transient, dt, min_freq2, numberofzones)
        # plot_spectrogram(ff, tt, Sxx)


    if args.analysis == 'interareal':
        dt = args.dt
        tstop = 40
        transient = 5
        # speciy number of areas that communicate with each other
        Nareas = 2
        t = np.arange(dt, tstop + dt - transient, dt)

        # define interlaminar synaptic coupling strenghts
        J_2e = 1; J_2i = 0
        J_5e = 0; J_5i = 0.75

        J = np.array([[wee, wie, J_5e,   0],
                      [wei, wii, J_5i,   0],
                      [J_2e, 0,   wee, wie],
                      [J_2i, 0,   wei, wii]])


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
        interareal_simulation(t, dt, tstop, J, W, tau, Iext, Ibkg, sig, args.sigmaoverride)
        # microstimulation


    if not args.nogui:
        plt.show()

