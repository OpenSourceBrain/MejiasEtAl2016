import os
import numpy as np
import matplotlib.pylab as plt
from scipy import signal
import math

from calculate_rate import calculate_rate


def debug_firing_rate(analysis, t, dt, tstop, J, tau, sig, Iexts, Ibgk, nruns, sigmaoverride, Nareas, noconns, initialrate):
    # calculate the firing rate
    for i in Iexts:
        # inject current only on excitatory layer
        Iext = np.array([i, 0, i, 0])

        for nrun in range(nruns):
            rate = calculate_rate(t, dt, tstop, J, tau, sig, Iext, Ibgk, sigmaoverride, Nareas, initialrate=initialrate)
            filename = os.path.join(analysis, \
                       'simulation_Iext%s_nrun%s_noise%s_dur%s%s_dt%s'%(i,nrun,sigmaoverride,t[-1],('_noconns' if noconns else ''),dt))

            # select the excitatory and inhibitory layers for L2/3
            uu_p_l2_3 = np.expand_dims(rate[0, :, 0], axis=1)
            vv_p_l2_3 = np.expand_dims(rate[1, :, 0], axis=1)
            # select the excitatory and inhibitory layers for L5/6
            uu_p_l5_6 = np.expand_dims(rate[2, :, 0], axis=1)
            vv_p_l5_6 = np.expand_dims(rate[3, :, 0], axis=1)

            # Plot the layers time course for layer L2/3
            plt.figure()
            plt.plot(uu_p_l2_3, label='excitatory', color='r')
            plt.plot(vv_p_l2_3, label='inhibitory', color='b')
            # plt.ylim([-.5, 2])
            plt.ylim(-.5, 5.5)
            plt.legend()
            plt.title('Layer 2/3; noise=%s; conns=%s'%(sigmaoverride,not noconns))
            plt.savefig(filename + '_L2_3.png')

            # Plot the layers time course for layer L5/6
            plt.figure()
            plt.plot(uu_p_l5_6, label='excitatory', color='r')
            plt.plot(vv_p_l5_6, label='inhibitory', color='b')
            plt.ylim(-.5, 5.5)
            plt.legend()
            plt.title('Layer 5/6; noise=%s; conns=%s'%(sigmaoverride,not noconns))
            plt.savefig(filename + '_L5_6.png')

            # save the simulation as a txt file and figure
            activity = np.concatenate((uu_p_l2_3, vv_p_l2_3, uu_p_l5_6, vv_p_l5_6), axis=1)
            np.savetxt(filename + '.txt', activity)
            plt.close('all')

        print('Saved debug info to %s.txt!'%filename )

def get_network_configuration(analysis_type, noconns=False):

    # Paper defined connectivity
    JEE = 1.5; JEI = -3.25;
    JIE = 3.5; JII = -2.5

    # Connection between layers
    if noconns:
        wee = 0.0; wei = 0.0
        wie = 0.0; wii = 0.0

    else:
        wee = JEE; wei = JIE
        wie = JEI; wii = JII

    # Specify membrane time constants
    tau_2e = 0.006; tau_2i = 0.015
    tau_5e = 0.030; tau_5i = 0.075
    tau = np.array([tau_2e, tau_2i, tau_5e, tau_5i])

    # sigma
    sig_2e = .3; sig_2i = .3
    sig_5e = .45; sig_5i = .45
    sig = np.array([sig_2e, sig_2i, sig_5e, sig_5i])

    if analysis_type == 'intralaminar':
        # define intralaminar synaptic coupling strenghts
        J_2e = 0; J_2i = 0
        J_5e = 0; J_5i = 0

        Imin = 0; Istep = 2; Imax = 6
        # Note: the range function does not include the end
        Iexts = range(Imin, Imax + Istep, Istep)


    elif analysis_type == 'interlaminar_a':
        # define interlaminar synaptic coupling strengths
        J_2e = 1; J_2i = 0
        J_5e = 0; J_5i = 0.75

        Iexts = np.array([8, 0, 8, 0])

    elif analysis_type == 'interlaminar_u':
        # Specifiy model where there is no interlaminar connection
        J_2e = 0; J_2i = 0
        J_5e = 0; J_5i = 0

        Iexts = np.array([8, 0, 8, 0])

    elif analysis_type == 'interlaminar_b':
        # define interlaminar synaptic coupling strengths
        J_2e = 1; J_2i = 0
        J_5e = 0; J_5i = 0.75

        Iexts = np.array([6, 0, 8, 0])

    elif analysis_type == 'debug_intralaminar':
        Iexts = [0]
        J_2e = 0; J_2i = 0
        J_5e = 0; J_5i = 0

    elif analysis_type == 'debug_interlaminar':
        Iexts = [0]
        J_2e = 0; J_2i = 0
        J_5e = 0; J_5i = 0

    elif analysis_type == 'interareal':
        # define interlaminar synaptic coupling strenghts
        J_2e = 1; J_2i = 0
        J_5e = 0; J_5i = 0.75

    else:
        raise Exception('This type of analysis is not implemented')
    J = np.array([[wee, wie, J_5e,   0],
                  [wei, wii, J_5i,   0],
                  [J_2e, 0,   wee, wie],
                  [J_2i, 0,   wei, wii]])


    Ibgk = np.zeros((J.shape[0]))
    return tau, sig, J, Iexts, Ibgk


def calculate_periodogram(re, transient, dt):
    """
    Does necessary preprocessing to the data and calculates periodogram
    Inputs
        re: Signal to be analysed
        transient: Number of discarted time points
        dt: time

    Returns:
        pxx: Power spectral density
        fxx: Array of sample frequencies
    """
    # calculate fft and sampling frequency for the peridogram
    fs = 1 / dt
    N = re.shape[0]
    win = signal.get_window('boxcar', N)
    # Calculate fft (number of freq. points at which the psd is estimated)
    # Calculate the max power of 2 and find the maximum value
    pow2 = int(round(math.log(N, 2)))
    fft = max(256, 2 ** pow2)

    # discard the first points
    restate = re[int(round((transient + dt)/dt)) - 1:]

    # perform periodogram on restate
    fxx, pxx = signal.periodogram(restate, fs=fs, window=win[int(round((transient + dt)/dt)) - 1:],
                                    nfft=fft, detrend=False, return_onesided=True,
                                    scaling='density')
    # print('Done calculating Periodogram!')
    return pxx, fxx

def find_peak_frequency(fxx,pxx, min_freq, restate):
    # find the frequency of the oscillations
    z = np.where(fxx > min_freq)
    pxx_freq = pxx[z]
    fxx_freq = fxx[z]

    # Locate peaks in the spectrum
    loc = signal.find_peaks(pxx_freq)[0]
    pks = pxx_freq[loc]
    z = len(loc)
    # if there is at least one peak
    if z > 1:
        # find index of the highest peak
        z3 = np.argmax(pks)
        # find location in Hz of the higest peak
        frequency = fxx_freq[loc[z3]]
        # power of the peak in the spectrum
        amplitudeA = pxx_freq[loc[z3]]
    else:
        frequency = 0
        amplitudeA = 0

    # calculate excitatory mean firing rate and the amplitute of oscillations
    mfr = np.mean(restate)
    amplitudeB = 2 * np.std(restate)
    amplitudeC = np.max(restate) - np.min(restate)
    return frequency, amplitudeA, amplitudeB, amplitudeC


def compress_data(pxx, fxx, bin):
    """
    Compress data, using a sliding window approach
    Input:
        bin: Pick one point every 'bin' points
        re: Data to be shrunken

    Output:
        pxx_bin:
        fxx_bin:
    """
    # We start by calculating the number of index needed to in order to sample every 5 points and select only those
    remaining = fxx.shape[0] - (fxx.shape[0] % bin)
    fxx2 = fxx[0:remaining]
    pxx2 = pxx[0:remaining]
    # Then we calculate the average signal inside the specified non-overlapping windows of size bin-size.
    # Note: the output needs to be an np.array in order to be able to use np.where afterwards
    pxx_bin = np.asarray([np.mean(pxx2[i:i + bin]) for i in range(0, len(pxx2), bin)])
    fxx_bin = np.asarray([np.mean(fxx2[i:i + bin]) for i in range(0, len(fxx2), bin)])
    return pxx_bin, fxx_bin

def firing_rate_analysis(noconns=False,
                         testduration=1000, # ms
                         sigmaoverride=None,
                         initialrate=5,
                         dt=2e-4):
                         
    ########################################################################################################################
    #                                                      Intralaminar
    ########################################################################################################################
    tstop = testduration/1000. # sec
    t = np.linspace(0, tstop, tstop/dt)
    # speciy number of areas that communicate with each other
    Nareas = 1

    # For Fig2 the simulation is run 10 and the averate is taken as a signal. Just as a test,
    # here we just run the simulation once
    nruns = 1

    analysis = 'debug'
    
    level = 'intralaminar'
    analysis = os.path.join(analysis, level)
    # check if folder exists otherwise create it
    if not os.path.isdir(analysis):
        os.mkdir(analysis)

    # Calculate firing rate, save the results as a txt file and plot the firing rate over time for layer L2_3
    tau, sig, J, Iexts, Ibgk = get_network_configuration('debug_intralaminar', noconns=noconns)
    debug_firing_rate(analysis, t, dt, tstop, J,
                      tau, sig, Iexts, Ibgk, nruns, sigmaoverride, Nareas, noconns, initialrate)
                      
    print("Finished debug simulation Intralaminar of duration %s ms; conns removed: %s"%(testduration, noconns))

    ########################################################################################################################
    #                                                      Interlaminar
    ########################################################################################################################

    tstop = testduration/1000. # sec
    t = np.linspace(0, tstop, tstop/dt)
    transient = 5
    # speciy number of areas that communicate with each other
    Nareas = 1

    nruns = 1

    analysis = 'debug'
    
    level = 'interlaminar'
    analysis = os.path.join(analysis, level)
    # check if folder exists otherwise create it
    if not os.path.isdir(analysis):
        os.mkdir(analysis)

    tau, sig, J, Iexts, Ibgk = get_network_configuration('debug_interlaminar', noconns=noconns)
    debug_firing_rate(analysis, t, dt, tstop, J,
                      tau, sig, Iexts, Ibgk, nruns, sigmaoverride, Nareas, noconns, initialrate)

    print("Finished debug simulation Interlaminar of duration %s ms; conns removed: %s"%(testduration, noconns))
