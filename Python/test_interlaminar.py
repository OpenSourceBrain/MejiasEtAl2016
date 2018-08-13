import numpy as np
import os
import pickle
import matplotlib.pylab as plt

from helper_functions import calculate_periodogram

analysis = 'interlaminar'
dt = 2e-04
transient = 5
min_freq5 = 4 # alpha range

print('Loading the simulated data')
picklename = os.path.join(analysis, 'simulation.pckl')
with open(picklename, 'rb') as filename:
    rate = pickle.load(filename)

print('Calculating Periodogram')
x = rate[2, :]
pxx, fxx = calculate_periodogram(x, transient, dt)

print('Plotting')
fig = plt.figure()
plt.plot(rate[2, 0:int(1/dt)], label='L5/6')
plt.plot(rate[0, 0:int(1/dt)], label='L2/3')
plt.legend()

plt.figure()
plt.loglog(fxx, pxx)
plt.xlabel('Frequency(Hz)')
plt.ylabel('L5/6 Power')
plt.xlim([1, 100])

plt.figure()
plt.xlabel('Frequency(Hz)')
plt.ylabel('L5/6 Power')
plt.semilogy(fxx, pxx)
plt.xlim([1, 100])

plt.figure()
plt.xlabel('Frequency(Hz)')
plt.ylabel('L5/6 Power')
plt.semilogx(fxx, pxx)
plt.ylim([0, 2000])
plt.show()