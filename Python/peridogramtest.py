from __future__ import division

import matplotlib.pylab as plt
from scipy import  signal
import scipy.io.matlab as scmat
import math
import numpy as np

matfile = '../Matlab/fig2/peridogram.mat'
dt = 2e-04
min_freq = 10
m_file = scmat.loadmat(matfile)
restate = m_file['restate']
restate = np.transpose(restate)

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

# Todo: look at the jupyternotebook and find how to start from 0
plt.figure()
plt.semilogy(fxx_bin, pxx_bin)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V**2/Hz)')
plt.xlim(0, max(fxx_bin))
plt.show()
print('hello')