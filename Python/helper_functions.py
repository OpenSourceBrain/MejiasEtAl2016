import os
import numpy as np
import matplotlib.pylab as plt
from scipy import signal
import math

from calculate_rate import calculate_rate


def debug_firing_rate(analysis, t, dt, tstop, J, tau, sig, Iexts, Ibgk, nruns, noise, Nareas, noconns, initialrate):
    # calculate the firing rate
    for i in Iexts:
        # inject current only on excitatory layer
        Iext = np.array([[i], [0], [i], [0]])

        for nrun in range(nruns):
            rate = calculate_rate(t, dt, tstop, J, tau, sig, Iext, Ibgk, noise, Nareas, initialrate)
            filename = os.path.join(analysis, \
                       'simulation_Iext%s_nrun%s_noise%s_dur%s%s'%(i,nrun,noise,t[-1],('_noconns' if noconns else '')))

            # select only the excitatory and inhibitory layers for L2/3
            uu_p_l2_3 = np.expand_dims(rate[0, :, 0], axis=1)
            vv_p_l2_3 = np.expand_dims(rate[1, :, 0], axis=1)
            # select only the excitatory and inhibitory layers for L5/6
            uu_p_l5_6 = np.expand_dims(rate[2, :, 0], axis=1)
            vv_p_l5_6 = np.expand_dims(rate[3, :, 0], axis=1)

            # Plot the layers time course for layer L2/3
            plt.figure()
            plt.plot(uu_p_l2_3, label='excitatory', color='r')
            plt.plot(vv_p_l2_3, label='inhibitory', color='b')
            # plt.ylim([-.5, 2])
            plt.ylim(-.5, 5.5)
            plt.legend()
            plt.title('Layer 2/3; noise=%s; conns=%s'%(noise,not noconns))
            plt.savefig(filename + '_L2_3.png')

            # Plot the layers time course for layer L5/6
            plt.figure()
            plt.plot(uu_p_l5_6, label='excitatory', color='r')
            plt.plot(vv_p_l5_6, label='inhibitory', color='b')
            plt.ylim(-.5, 5.5)
            plt.legend()
            plt.title('Layer 5/6; noise=%s; conns=%s'%(noise,not noconns))
            plt.savefig(filename + '_L5_6.png')

            # save the simulation as a txt file and figure
            activity = np.concatenate((uu_p_l2_3, vv_p_l2_3, uu_p_l5_6, vv_p_l5_6), axis=1)
            np.savetxt(filename + '.txt', activity)
            plt.close('all')

        print('Saved debug info to %s!'%filename)


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
    sq_data = np.squeeze(restate)
    fxx, pxx = signal.periodogram(sq_data, fs=fs, window=win[int(round((transient + dt)/dt)) - 1:],
                                    nfft=fft, detrend=False, return_onesided=True,
                                    scaling='density')
    print('Done calculating Periodogram!')
    return pxx, fxx


def compress_data(pxx, fxx, bin):
    """
    Compress data
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
                         noise = 1,
                         initialrate=5): 
                         
    ########################################################################################################################
    #                                                      Intralaminar
    ########################################################################################################################
    # Connection between layers
    wee = 1.5; wei = -3.25
    wie = 3.5; wii = -2.5
    
    if noconns:
        wee = 0; wei = 0
        wie = 0; wii = 0


    # Specify membrane time constants
    tau_2e = 0.006; tau_2i = 0.015
    tau_5e = 0.030; tau_5i = 0.075
    tau = np.array([[tau_2e], [tau_2i], [tau_5e], [tau_5i]])

    # sigma
    sig_2e = .3; sig_2i = .3
    sig_5e = .45; sig_5i = .45
    sig = np.array([[sig_2e], [sig_2i], [sig_5e], [sig_5i]])

    dt = 2e-4 # sec
    tstop = testduration/1000. # sec
    t = np.linspace(0, tstop, tstop/dt)
    # speciy number of areas that communicate with each other
    Nareas = 1

    # define intralaminar synaptic coupling strenghts
    J_2e = 0; J_2i = 0
    J_5e = 0; J_5i = 0

    J = np.array([[wee, wei, J_5e,   0],
                  [wie, wii, J_5i,   0],
                  [J_2e, 0,   wee, wei],
                  [J_2i, 0,   wie, wii]])

    Iexts = [0]
    Ibgk = np.zeros((J.shape[0], 1))
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
    debug_firing_rate(analysis, t, dt, tstop, J,
                      tau, sig, Iexts, Ibgk, nruns, noise, Nareas, noconns, initialrate)
                      
    print("Finished debug simulation Intralaminar of duration %s ms; conns removed: %s"%(testduration, noconns))

    ########################################################################################################################
    #                                                      Interlaminar
    ########################################################################################################################

    # Define dt and the trial length
    dt = 2e-4
    tstop = testduration/1000. # sec
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
    Ibgk = np.zeros((J.shape[0], 1))
    nruns = 1

    analysis = 'debug'
    
    level = 'interlaminar'
    analysis = os.path.join(analysis, level)
    # check if folder exists otherwise create it
    if not os.path.isdir(analysis):
        os.mkdir(analysis)

    debug_firing_rate(analysis, t, dt, tstop, J,
                      tau, sig, Iexts, Ibgk, nruns, noise, Nareas, noconns, initialrate)

    print("Finished debug simulation Interlaminar of duration %s ms; conns removed: %s"%(testduration, noconns))
