from __future__ import print_function, division

import numpy as np
import math
import matplotlib.pylab as plt
import argparse
from scipy import signal
import pickle
import scipy.io.matlab as scmat
# set random set
#np.random.seed(42)


def transduction_function(x):
    # note: define boundary conditions for the transduction function
    if x == 0:
        return 1
    elif x <= -100:
        return 0
    else:
        return x / (1 - math.exp(-x))


def calculate_firing_rate(dt, re, ri, wee, wie, wei, wii, tau_e, tau_i, sei, xi_i, xi_e, Iext_e, Iext_i):
    tstep2e = ((dt * sei * sei) / tau_e) ** .5
    tstep2i = ((dt * sei * sei) / tau_i) ** .5
    dE = dt * (-re + transduction_function((wee * re) + (wei * ri) + Iext_e) + tstep2e * xi_e)/tau_e
    dI = dt * (-ri + transduction_function((wie * re) + (wii * ri) + Iext_i) + tstep2i * xi_i)/tau_i
    uu_p = re + dE
    vv_p = ri + dI
    return uu_p, vv_p


def calculate_rate(t, dt, tstop, wee, wie, wei, wii, tau_e, tau_i, sei, Iext_e, Iext_i, plot=False):
    uu_p = np.zeros((len(t) + 1, 1))
    vv_p = np.zeros((len(t) + 1, 1))

    mean_xi = 0
    std_xi = 1
    xi_e = np.random.normal(mean_xi, std_xi, int(round(tstop/dt)) + 1)
    xi_i = np.random.normal(mean_xi, std_xi, int(round(tstop/dt)) + 1)

    # Initial rate values
    # Note: the 5 ensures that you have between 0 and 10 spikes/s
    uu_p[0] = 5 * (1 + np.tanh(2 * xi_e[0]))
    vv_p[0] = 5 * (1 + np.tanh(2 * xi_i[0]))

    for dt_idx in range(len(t)):
        uu_p[dt_idx + 1], vv_p[dt_idx + 1] = calculate_firing_rate(dt, uu_p[dt_idx], vv_p[dt_idx], wee, wie, wei, wii,
                                                                   tau_e, tau_i, sei, xi_i[dt_idx + 1], xi_e[dt_idx + 1],
                                                                   Iext_e, Iext_i)

    if plot:
        tplot = np.linspace(0, tstop, tstop/dt + 1)
        fig = plt.figure()
        plt.plot(tplot, vv_p, label='vv', color='blue')
        plt.legend()
        plt.title(args.layer)
        plt.ylim(0.6, 0.8)
        plt.xlabel('Time')
        plt.ylabel('Proportion of firing cells')
        plt.savefig('E_activity.png')

        fig = plt.figure()
        plt.plot(tplot, uu_p, label='uu', color='red')
        plt.legend()
        plt.title(args.layer)
        plt.ylim(0.2, 0.6)
        plt.xlabel('Time')
        plt.ylabel('Proportion of firing cells')
        plt.savefig('I_activity.png')
        plt.show()

        fig = plt.figure()
        plt.plot(tplot, abs(uu_p - vv_p), label='sum', color='green')
        plt.title(args.layer)
        plt.ylabel('Proportion of firing cells')
        plt.xlabel('Time')
        plt.legend()
        plt.show()

    return uu_p, vv_p


def down_sampled_periodogram(restate, fft, fs, win, min_freq):
    # perform periodogram on restate
    fxx2, pxx2 = signal.periodogram(np.squeeze(restate), nfft=fft, fs=fs, window=win, return_onesided=True,
                                    scaling='density', detrend=False)

    # Compress the data by sampling every 5 points.
    bin_size = 5
    # We start by calculating the number of index needed to in order to sample every 5 points and select only those
    remaining = fxx2.shape[0] - (fxx2.shape[0] % bin_size)
    fxx2 = fxx2[0:remaining]
    pxx2 = pxx2[0:remaining]
    # Then we calculate the average signal inside the specified bins
    # Note: the output needs to be an np.array in order to be able to use np.where afterwards
    pxx_bin = np.asarray([np.mean(pxx2[i:i+bin_size]) for i in range(0, len(pxx2), bin_size)])
    fxx_bin = np.asarray([np.mean(fxx2[i:i+bin_size]) for i in range(0, len(fxx2), bin_size)])

    # find the frequency of the oscillations (note used at the moment)
    z = np.where(fxx_bin > min_freq)
    pxx_freq = pxx_bin[z]
    fxx_freq = fxx_bin[z]

    return pxx_bin, fxx_bin


parser = argparse.ArgumentParser(description='Parameters for the simulation')
parser.add_argument('-tau_e', type=float, dest='tau_e', help='Excitatory membrane time constant (tau_e)')
parser.add_argument('-tau_i', type=float, dest='tau_i', help='Inhibitory membrane time constant (tau_i)')
parser.add_argument('-sei',   type=float, dest='sei', help='Deviation for the Gaussian white noise (s_ei)')
parser.add_argument('-layer', type=str, dest='layer', help='Layer of interest')
parser.add_argument('-nogui', dest='nogui', action='store_true',  help='No gui')

args = parser.parse_args()

## superficial layer
wee = 1.5
wei = -3.25
wie = 3.5
wii = -2.5

dt = 2e-4
tstop = 25
transient = 5

t = np.linspace(0, tstop, tstop/dt)

# Iterate over different input strength
Imin = 0
Istep = 2
Imax = 6
# Note: the range function does not include the end
Iexts = range(Imin, Imax + Istep, Istep)
psd_dic = {}

dt = 2e-04
min_freq = 10
# sampling frequency to calculate the peridogram
fs = 1/dt

for Iext in Iexts:

    psd_dic[Iext] = {}
    psd_dic[Iext]['pxx'] = []
    # run each combination of external input multiple times an take the average PSD
    nruns = 10

    for nrun in range(nruns):


        # inject current only on excitatory layer
        Iext_e = Iext * 1
        Iext_i = 0
        uu_p, vv_p = calculate_rate(t, dt, tstop, wee, wie, wei, wii, args.tau_e, args.tau_i, args.sei, Iext_e,
                                    Iext_i, plot=False)


        # matfile = '../Matlab/fig2/peridogram.mat'
        # m_file = scmat.loadmat(matfile)
        # restate = np.transpose(m_file['restate'])
        restate = uu_p
        # discard the first time points
        # lower_boundary = int((round(transient + dt) / dt))
        # restate = restate[lower_boundary:-1]

        # define window to use for the periodogram calculation
        N = restate.shape[0]
        win = signal.get_window('boxcar', N)
        # Calculate fft (number of freq. points at which the psd is estimated)
        # Calculate the max power of 2 and find the maximum value
        pow2 = int(round(math.log(N, 2)))
        fft = max(256, 2 ** pow2)

        # perform periodogram on restate. As the fxx_bin is constant for all runs we do not save it
        pxx_bin, fxx_bin = down_sampled_periodogram(restate, fft, fs, win, min_freq)

        window_size = 81
        mask = np.ones(81, 'd')
        s = np.r_[pxx_bin[window_size-1:0:-1], pxx_bin, pxx_bin[-2:window_size-1:1]]
        pxx = np.convolve(mask/mask.sum(), s, mode='valid')

        psd_dic[Iext]['pxx'].append(pxx)
        # transform array into numpy array
    # take the mean and std over the different runs
    psd_dic[Iext]['pxx'] = np.array(psd_dic[Iext]['pxx'])
    psd_dic[Iext]['mean_pxx'] = np.mean(psd_dic[Iext]['pxx'], axis=0)
    psd_dic[Iext]['std_pxx'] = np.std(psd_dic[Iext]['pxx'], axis=0)

# add fxx_bin to dictionary
psd_dic['fxx_bin'] = fxx_bin

for Iext in Iexts:
    fig = plt.figure()
    plt.semilogy(fxx_bin, psd_dic[Iext]['mean_pxx'])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (V**2/Hz)')
    plt.xlim(0, max(fxx_bin))


if not args.nogui:
    plt.show()

# save the results into a pickle file
with open('intralaminar_simulation.pckl', 'w') as file:
    file.write(pickle.dumps(psd_dic))

print('Done')



