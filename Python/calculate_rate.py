#!/usr/bin/env python
from __future__ import print_function, division

import numpy as np
import math
from numba import jit, vectorize
# set random set
np.random.seed(42)

@vectorize
def transduction_function(element):
    if element == 0:
        return 1
    elif element <= -100:
        return 0
    else:
        return element / (1 - math.exp(-element))

@vectorize(nopython=True)
def numba_calculate_transfer_input(rate, Ibgk, Iext):
    wee = 1.5; wei = -3.25
    wie = 3.5; wii = -2.5
    J_2e = 0; J_2i = 0
    J_5e = 0; J_5i = 0
    J = np.array([[wee, wei, J_5e, 0],
                  [wie, wii, J_5i, 0],
                  [J_2e, 0, wee, wei],
                  [J_2i, 0, wie, wii]])
    # elementwise add elements in Ibgk, Iext, tmp
    total_input = Ibgk + Iext + np.dot(J, rate)
    # calculate input after the transfer function
    transfer_input = transduction_function(total_input)
    return transfer_input

@vectorize
def numba_calculate_rate(rate, transfer_input, tau_r, tstep2, xi):
    delta_rate = tau_r * (rate + transfer_input) + tstep2 * xi
    return delta_rate


@jit(nopython=True)
def dt_calculate_rate(J, rate, Ibgk, Iext, dt, tau, tstep2, xi):
    # calculate total input current
    tmp = np.dot(J, rate)
    # elementwise add elements in Ibgk, Iext, tmp
    total_input = np.add(Ibgk, np.add(Iext, tmp))
    # calculate input after the transfer function
    transfer_input = transduction_function(total_input)
    tau_r = np.divide(dt, tau)
    delta_rate = np.add(np.multiply(tau_r, (np.add(-rate, transfer_input))),
                        np.multiply(tstep2, xi))
    return delta_rate


def calculate_rate(t, dt, tstop, J, tau, sig, Iext, Ibgk, noise, Nareas, W=1, Gw=1):
    """
    Calculates the region rate over time

    :param t:
    :param dt:
    :param tstop:
    :param J:
    :param tau: Membrane tiem constant
    :param sig:
    :param Iext: Additional current
    :param Ibgk: Background current of the system
    :param noise:
    :param Nareas: Number of Areas to take into account
    :param W: Intraareal connectivity matrix
    :param Gw:
    :return:
        rate: Rate over time for the areas of interest
        mean_input:
    """


    rate = np.zeros((4, int(round(tstop/dt) + 1), Nareas))
    # Apply additional input current only on excitatory layers
    tstep2 = ((dt * sig * sig) / tau) ** .5
    mean_xi = 0
    std_xi = noise
    xi = np.random.normal(mean_xi, std_xi, (4, int(round(tstop/dt)) + 1, Nareas))

    # Initial rate values
    # Note: the 5 ensures that you have between 0 and 10 spikes/s
    rate[:, 0, :] = 5 * (1 + np.tanh(2 * xi[:, 0, :]))

    for dt_idx in range(len(t)):
        # iterate over different areas. Only true for the interareal simulation
        for area in range(Nareas):
            tau_r = np.divide(dt, tau)
            transfer_input = numba_calculate_transfer_input(J, rate[:, dt_idx, area], Ibgk, Iext)
            delta_rate = numba_calculate_rate(rate[:, dt_idx, area], transfer_input, tau_r, tstep2, xi[:, dt_idx, area])
            # delta_rate = dt_calculate_rate(J, rate[:, dt_idx, area], Ibgk, Iext, dt, tau, tstep2, xi[:, dt_idx, area])
            rate[:, dt_idx + 1, area] = np.add(rate[:, dt_idx, area], delta_rate)

    # exclude the initial point that corresponds to the initial conditions
    rate = rate[:, 1:, :]

    return rate
