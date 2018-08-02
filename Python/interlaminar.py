import numpy as np
import os
import pickle
from scipy import signal, fftpack
import matplotlib.pylab as plt

from calculate_rate import calculate_rate
from helper_functions import calculate_periodogram


def interlaminar_simulation(analysis, t, dt, tstop, J, tau, sig, Iext_a, noise):
    rate = calculate_rate(t, dt, tstop, J, tau, sig, Iext_a, noise)
    picklename = os.path.join(analysis, 'simulation.pckl')
    with open(picklename, 'wb') as filename:
        pickle.dump(rate, filename)
    print('Done Simulation!')


def find_peak_frequency(fxx,pxx, min_freq):
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
        amplitude = pxx_freq[loc[z3]]
    else:
        frequency = 0
        amplitude = 0
    return frequency

def my_pretransformations(x, window, noverlap, fs):

    # Place x into columns and return the corresponding central time estimates
    # restructure the data
    ncol = int(np.floor((x.shape[0] - noverlap) / (window.shape[0] - noverlap)))
    coloffsets = np.expand_dims(range(ncol), axis=0) * (window.shape[0] - noverlap)
    rowindices = np.expand_dims(range(0, window.shape[0]), axis=1)

    # segment x into individual columns with the proper offsets
    xin = x[rowindices + coloffsets]
    # return time vectors
    t = coloffsets + (window.shape[0]/2)/ fs
    return xin, t


def interlaminar_activity_analysis(rate, transient, dt, t, min_freq5):

    # Note: This analysis selects only the excitatory populations from L2/3 and L5/6
    x_2 = rate[0, int(round((transient + dt)/dt)) - 1:]
    x_5 = rate[2, int(round((transient + dt)/dt)) - 1:]

    pxx, fxx = calculate_periodogram(x_5, transient, dt)
    f_peakalpha = find_peak_frequency(fxx, pxx, min_freq5)
    print('Average peak frequency on the alpha range: %.02f Hz' %f_peakalpha)

    # band-pass filter L5 activity
    fmin = 7; fmax = 12; fs = 1/dt
    filter_order = 3
    bf, af = signal.butter(filter_order, [fmin/(fs/2), fmax/(fs/2)], 'bandpass')
    # Note: padlen is differently defined in the scipy implementation
    # simulated LFP
    re5bp = -signal.filtfilt(bf, af, x_5, padlen=3*(max(len(af), len(bf)) - 1))


    # Locate N well spaced peaks along the trial
    tzone = 4
    # length of the tzone in indices
    tzoneindex = int(round((tzone/dt)))
    rest = len(t) % tzoneindex
    time5 = t[0:-rest]; re5 = re5bp[0:-rest]; re2 = x_2[0:-rest]
    numberofzones = int(round(len(re5)/tzoneindex))
    zones5 = np.reshape(re5, (tzoneindex, numberofzones), order='F')
    zones2 = np.reshape(re2, (tzoneindex, numberofzones), order='F')

    # find a prominent peak around the center of each zone
    # Note: slicing in matlab includes the last index
    tzi_bottom = int(round(tzoneindex/2-tzoneindex/4)) + 1
    tzi_top = int(round(tzoneindex/2+tzoneindex/4)) + 1

    alpha_peaks = np.zeros((numberofzones))
    aploc = np.zeros((numberofzones))
    # find max value for each zone
    for i in range(numberofzones):
        alpha_peaks[i] = np.max(zones5[tzi_bottom:tzi_top, i])
        # Todo: indices are shifted by two respect to matlab code
        aploc[i] = np.argmax(zones5[tzi_bottom:tzi_top, i]) + tzi_bottom

    # chose a segment of 7 cycles centered on the prominent peak of each zone
    seglength = 7/f_peakalpha
    # Check if there is any problems with the segment window
    if seglength/2 >= tzi_bottom * dt:
        print('Problems with segment window!')

    # segment semi-length in indices
    segindex = int(round(0.5 * seglength/dt))
    # Note: The + 2 corrects for indexing in python
    segind1 = int(round(aploc[0] - segindex) + 2)
    segind2 = int(round(aploc[0] + segindex) + 2)
    # Note: + 1 correct for inclusive range in matlab
    segment2 = np.zeros((segind2 - segind1 + 1, numberofzones))
    segment5 = np.zeros((segind2 - segind1 + 1, numberofzones))
    for i in range(numberofzones):
        if alpha_peaks[i] >= 0:
            segment5[:, i] = zones5[segind1:segind2 + 1, i]
            segment2[:, i] = zones2[segind1:segind2 + 1, i]
    return segment5, segindex, numberofzones

def interlaminar_analysis_periodeogram(rate, transient, dt, min_freq2, numberofzones):
    # TODO: still in construction
    # calculate the spectogram for L2/3 and average the results over the segments
    pxx2, fxx2 = calculate_periodogram(rate[0, :], transient, dt)
    f_peakgamma = find_peak_frequency(fxx2, pxx2, min_freq2)
    print('Average peak frequency on the gamma range: %.02f Hz' %f_peakgamma)
    timewindow = 7/f_peakgamma
    window_len = int(round(timewindow/dt))
    window = signal.get_window('hamming', window_len)
    noverlap = int(round(0.95 * window_len))


    # try loading mat file with the correct input to the spectogram
    from scipy.io import loadmat
    matfile = '../Matlab/fig3/segment2.mat'
    mat = loadmat(matfile)
    segment2 = mat['segment2']

    # calculate nfft
    lowest_frequency = 25; highest_frequency = 45; step_frequency = .25
    freq_displayed = np.arange(lowest_frequency, highest_frequency + step_frequency, step_frequency)
    fs = 1/dt

    xin, t = my_pretransformations(segment2[:, 0], window, noverlap, fs)
    data = np.multiply(np.expand_dims(window, axis=1), xin)
    # TODO: Try to use fft instead of the goertzel algorithm to calculate the fft
    Xx = np.zeros((freq_displayed.shape[0], xin.shape[1]), dtype=complex)
    for i in range(xin.shape[1]):
        Xx[:, i] = fftpack.fft(data[:, i], freq_displayed.shape[0])


    nfft = np.arange(lowest_frequency, highest_frequency + step_frequency, step_frequency)

    # obtain spectograms
    for i in range(numberofzones):
        # first one is the one working the best
        ff, tt, Sxx = signal.spectrogram(segment2[:, i], fs=fs, window=window, noverlap=noverlap, return_onesided=True,
                                         detrend=False, scaling='density', mode='psd')
        # try to get only the frequencies between 25 and 45
        ff, tt, Sxx = signal.spectrogram(segment2[:, i], fs=fs, nfft=nfft, return_onesided=True, detrend=False, scaling='density',
                                         mode='psd')
        ff, tt, Sxx = signal.spectrogram(segment2[:, i], fs=fs, return_onesided=True, detrend=False,
                                         scaling='density', mode='psd')

    print('Done Analysis!')


def my_spectrogram(x, window, noverlap, f, fs):
    # Process the frequency-specific arguments
    nfft = f

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def plot_activity_traces(dt, segment5, segindex):
    # calculate the peak-centered alpha wave by averaging
    alphawaves = np.mean(segment5, axis=1)
    alphatime = [(i*dt) - (segindex*dt) for i in range(1, alphawaves.shape[0] + 1)]
    plt.figure()
    # plot the first 100 elements from segment5
    grey_rgb = (.7, .7, .7)
    plt.plot(alphatime, segment5[:, 0:100], color=grey_rgb)
    plt.plot(alphatime, alphawaves, 'b')
    plt.xlabel('Time relative to alpha peak (s)')
    plt.ylabel('LFP, L5/6')
    plt.xlim([-.24, .24])
