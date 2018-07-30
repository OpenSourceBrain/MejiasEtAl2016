from __future__ import print_function, division

import numpy as np
import math
import matplotlib.pylab as plt
# set random set
np.random.RandomState(seed=42)


def transduction_function(x):
    # note: define boundary conditions for the transduction function
    # Calculate the transduction function for each layer separately. This assumes that each layer is one
    # row in the passed input
    x_transducted = np.zeros((x.shape))
    for idx, element in enumerate(x):
        if element == 0:
            x_transducted[idx] = 1
        elif element <= -100:
            x_transducted[idx] = 0
        else:
            x_transducted[idx] = element / (1 - math.exp(-element))
    return x_transducted


def calculate_rate(t, dt, tstop, J, tau, sig, Iext, noise):

    rate = np.zeros((4, len(t) + 1))
    # Apply additional input current only on excitatory layers
    tstep2 = ((dt * sig * sig) / tau) ** .5
    mean_xi = 0
    std_xi = noise
    xi = np.random.normal(mean_xi, std_xi, (4, int(round(tstop/dt)) + 1))

    # Initial rate values
    # Note: the 5 ensures that you have between 0 and 10 spikes/s
    rate[:, 0] = 5 * (1 + np.tanh(2 * xi[:, 0]))

    for dt_idx in range(len(t)):

        # calculate total input current
        total_input = Iext + np.expand_dims(np.dot(J, rate[:, dt_idx]), axis=1)
        # calculate input after the transfer function
        transfer_input = transduction_function(total_input)
        delta_rate = dt / tau * (- np.expand_dims(rate[:, dt_idx], axis=1) + transfer_input) + \
                                 tstep2 * np.expand_dims(xi[:, dt_idx], axis=1)
        rate[:, dt_idx + 1] = np.squeeze(np.expand_dims(rate[:, dt_idx], axis=1) + delta_rate)

    return rate
