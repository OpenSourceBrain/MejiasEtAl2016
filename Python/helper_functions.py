import os
import numpy as np
import matplotlib.pylab as plt
from scipy import signal
import math

from calculate_rate import calculate_rate


def debug_neuroml(analysis, layer, t, dt, tstop, J, tau, sig, Iexts, nruns, noise, nogui):
    # calculate the firing rate
    for i in Iexts:
        # inject current only on excitatory layer
        Iext = np.array([[i], [0], [i], [0]])

        for nrun in nruns:
            rate = calculate_rate(t, dt, tstop, J, tau, sig, Iext, noise)

            # select only the excitatory and inhibitory layers for L2/3
            uu_p = np.expand_dims(rate[0, :], axis=1)
            vv_p = np.expand_dims(rate[1, :], axis=1)
            # Plot the layers time course
            plt.figure()
            plt.plot(uu_p, label='excitatory', color='r')
            plt.plot(vv_p, label='inhibitory', color='b')
            plt.ylim([-.5, 2])
            plt.legend()
            plt.title('noise=' + str(noise))

            # save the simulation as a txt file
            filename = os.path.join(analysis, layer + '_simulation_Iext_'+
                                    str(i) + '_nrun_' + str(nrun))
            activity = np.concatenate((uu_p, vv_p), axis=1)
            np.savetxt(filename + '.txt', activity)

    print('Done debugging!')


def down_sampled_periodogram(re, transient, dt):
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
    fxx2, pxx2 = signal.periodogram(sq_data, fs=fs, window=win[int(round((transient + dt)/dt)) - 1:],
                                    nfft=fft, detrend=False, return_onesided=True,
                                    scaling='density')

    # Compress the data by sampling every 5 points.
    bin_size = 5
    # We start by calculating the number of index needed to in order to sample every 5 points and select only those
    remaining = fxx2.shape[0] - (fxx2.shape[0] % bin_size)
    fxx2 = fxx2[0:remaining]
    pxx2 = pxx2[0:remaining]
    # Then we calculate the average signal inside the specified non-overlapping windows of size bin-size.
    # Note: the output needs to be an np.array in order to be able to use np.where afterwards
    pxx_bin = np.asarray([np.mean(pxx2[i:i+bin_size]) for i in range(0, len(pxx2), bin_size)])
    fxx_bin = np.asarray([np.mean(fxx2[i:i+bin_size]) for i in range(0, len(fxx2), bin_size)])

    return pxx_bin, fxx_bin