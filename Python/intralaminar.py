from __future__ import print_function, division

import numpy as np
import math
import matplotlib.pylab as plt
import argparse
from scipy import signal
from scipy.fftpack import fft
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


def calculate_firing_rate(dt, re, ri, wee, wie, wei, wii, tau_e, tau_i, sei, xi_i, xi_e):
    tstep2e = ((dt * sei * sei) / tau_e) ** .5
    tstep2i = ((dt * sei * sei) / tau_i) ** .5
    dE = dt * (-re + transduction_function((wee * re) + (wei * ri)) + tstep2e * xi_e)/tau_e
    dI = dt * (-ri + transduction_function((wie * re) + (wii * ri)) + tstep2i * xi_i)/tau_i
    uu_p = re + dE
    vv_p = ri + dI
    return uu_p, vv_p

parser = argparse.ArgumentParser(description='Parameters for the simulation')
parser.add_argument('-tau_e', type=float, dest='tau_e', help='Excitatory membrane time constant (tau_e)')
parser.add_argument('-tau_i', type=float, dest='tau_i', help='Inhibitory membrane time constant (tau_i)')
parser.add_argument('-sei',   type=float, dest='sei', help='Deviation for the Gaussian white noise (s_ei)')
parser.add_argument('-layer', type=str, dest='layer', help='Layer of interest')

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
                                                               args.tau_e, args.tau_i, args.sei, xi_i[dt_idx + 1], xi_e[dt_idx + 1])

tplot = np.linspace(0, tstop, tstop/dt + 1)
plt.figure()
plt.plot(tplot, vv_p, label='vv', color='blue')
plt.legend()
plt.title(args.layer)
plt.ylim(0.6, 0.8)
plt.xlabel('Time')
plt.ylabel('Proportion of firing cells')
plt.savefig('E_activity.png')
plt.show()

plt.figure()
plt.plot(tplot, uu_p, label='uu', color='red')
plt.legend()
plt.title(args.layer)
plt.ylim(0.2, 0.6)
plt.xlabel('Time')
plt.ylabel('Proportion of firing cells')
plt.savefig('I_activity.png')
plt.show()

plt.figure()
plt.plot(tplot, abs(uu_p - vv_p), label='sum', color='green')
plt.title(args.layer)
plt.ylabel('Proportion of firing cells')
plt.xlabel('Time')
plt.legend()
plt.show()


#m_file = scmat.loadmat(matfile)
restate = uu_p
# discard the first time points
lower_boundary = int((round(transient + dt) / dt))
restate = restate[lower_boundary:-1]


N = restate.shape[0]
# sampling frequency to calculate the peridogram
fs = 1/dt
# define window to use for the periodogram calculation
win = signal.get_window('boxcar', N)
# Calculate fft (number of freq. points at which the psd is estimated)
# Calculate the max power of 2 and find the maximum value
pow2 = int(round(math.log(N, 2)))
fft = max(256, 2 ** pow2)

# perform periodogram on restate
# fxx, pxx = signal.periodogram(restate, nfft=fft, fs=fs, window=win, return_onesided=True,
#                               scaling='density', detrend=False)
fxx2, pxx2 = signal.periodogram(np.transpose(restate), nfft=fft, fs=fs, window=win, return_onesided=True,
                                scaling='density', detrend=False)


# The fancy plot is only for layer 2/3 and it is average over 10 rounds.
# TODO:smooth data and save it
# Todo: look at the jupyternotebook and find how to start from 0
plt.figure()
plt.semilogy(fxx2, np.squeeze(pxx2))
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V**2/Hz)')
plt.show()
print('hello')
