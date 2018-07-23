from __future__ import division

import matplotlib.pylab as plt
from scipy import  signal
import scipy.io.matlab as scmat
import math
import numpy as np

matfile = '../Matlab/fig2/peridogram.mat'
dt = 2e-04
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
# fxx, pxx = signal.periodogram(restate, nfft=fft, fs=fs, window=win, return_onesided=True,
#                               scaling='density', detrend=False)
fxx2, pxx2 = signal.periodogram(np.transpose(restate), nfft=fft, fs=fs, window=win, return_onesided=True,
                              scaling='density', detrend=False)


plt.figure()
plt.semilogy(fxx2, np.squeeze(pxx2))
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V**2/Hz)')
plt.show()
print('hello')