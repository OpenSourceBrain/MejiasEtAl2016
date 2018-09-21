#!/usr/bin/env python
from __future__ import print_function, division

import numpy as np
import math
import matplotlib.pylab as plt
# set random set
np.random.seed(42)


def transduction_function(element):
    if element == 0:
        return 1
    elif element <= -100:
        return 0
    else:
        return element / (1 - math.exp(-element))

def test_zip(dt, tau, rate, dt_idx, area, transfer_input, tstep2, xi):
    delta_rate = [dt / tau_region * (- region_rate + region_transfer_input) + region_tstep * region_initial_conditions for \
                  tau_region, region_rate, region_transfer_input, region_tstep, region_initial_conditions in \
                  zip(tau, rate[:, dt_idx, area], transfer_input, tstep2, xi[:, dt_idx, area])]
    rate[:, dt_idx + 1, area] = [previous_rate + region_rate for previous_rate, region_rate in
                                 zip(rate[:, dt_idx, area], delta_rate)]
    return rate

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
            # calculate total input current
            tmp = np.dot(J, rate[:, dt_idx, area])
            # elementwise add elements in Ibgk, Iext, tmp
            total_input = np.add(Ibgk, np.add(Iext, tmp))
            # calculate input after the transfer function
            transfer_input = map(transduction_function, total_input)
            tau_r = np.divide(dt, tau)
            delta_rate = np.add(np.multiply(tau_r, (np.add(-rate[:, dt_idx, area], transfer_input))),
                                np.multiply(tstep2, xi[:, dt_idx, area]))
            rate[:, dt_idx + 1, area] = np.add(rate[:, dt_idx, area], delta_rate)

    # exclude the initial point that corresponds to the initial conditions
    rate = rate[:, 1:, :]

    return rate
