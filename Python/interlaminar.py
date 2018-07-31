import numpy as np
import os
import pickle
from scipy import signal

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


def interlaminar_analysis(analysis, transient, dt, t, min_freq):
    # load data
    picklename = os.path.join(analysis, 'simulation.pckl')
    with open(picklename, 'rb') as filename:
        rate = pickle.load(filename)

    # Note: This analysis selects only the excitatory populations from L2/3 and L5/6
    x_2 = rate[0, int(round((transient + dt)/dt)) - 1:]
    x_5 = rate[2, int(round((transient + dt)/dt)) - 1:]

    pxx, fxx = calculate_periodogram(x_5, transient, dt)
    frequency = find_peak_frequency(fxx, pxx, min_freq)

    # band-pass filter L5 activity
    fmin = 7; fmax = 12; fs = 1/dt
    filter_order = 3
    bf, af = signal.butter(filter_order, [fmin/fs, fmax/fs], 'bandpass')
    # simulated LFP
    re5bp = signal.filtfilt(bf, af, x_5)
    # Locate N well spaced peaks along the trial
    tzone = 4
    # length of the tzone in indices
    tzoneindex = int(round((tzone/dt)))
    rest = len(t) % tzoneindex
    time5 = t[:-rest]; re5 = re5bp[:-rest]; re2 = x_2[:-rest]

    print('Done Analysis!')