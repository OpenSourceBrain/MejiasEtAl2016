#!/usr/bin/env/python
import os
import numpy as np
import pickle

from calculate_rate import calculate_rate

# model settings
dt = 2e-4
tstop = 1
t = np.arange(0, tstop, dt)
transient = 0
# Connection between layers
wee = 1.5; wei = -3.25
wie = 3.5; wii = -2.5
# define interlaminar synaptic coupling strenghts
J_2e = 0; J_2i = 0
J_5e = 0; J_5i = 0
J = np.array([[wee, wei, J_5e, 0],
                   [wie, wii, J_5i, 0],
                   [J_2e, 0, wee, wei],
                   [J_2i, 0, wie, wii]])
# Specify membrane time constants
tau_2e = 0.006; tau_2i = 0.015
tau_5e = 0.030; tau_5i = 0.075
tau = np.array([[tau_2e], [tau_2i], [tau_5e], [tau_5i]])

# sigma
sig_2e = .3; sig_2i = .3
sig_5e = .45; sig_5i = .45
sig = np.array([[sig_2e], [sig_2i], [sig_5e], [sig_5i]])

Iexts = [0]
Ibgk = np.zeros((J.shape[0], 1))
nruns = [1]
noise = 1
Nareas = 1

# calculate rate
rate = calculate_rate(t, dt, tstop, J, tau,
                      sig, Iexts, Ibgk, noise,
                      Nareas)
# Check if file already exists, if yes return an error, otherwise save the
# results
filename = os.path.join('..', 'tests','golden', 'simulation.pckl')
if os.path.isfile(filename):
    raise Exception("The golden data file already exists")
else:
    with open(filename, 'wb') as fh:
        pickle.dump(rate, fh)
    print('Saved golden results!')
