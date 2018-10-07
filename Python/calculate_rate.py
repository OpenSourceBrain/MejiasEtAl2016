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



def calculate_rate(t, dt, tstop, J, tau, sig, Iext, Ibgk, noise, Nareas, W=1, Gw=1, initialrate=-1):
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
    :param initialrate: Use this for t=0; if initialrate <0, use random value
    :return:
        rate: Rate over time for the areas of interest
        mean_input:
    """
    #print('Calculating rates for %i area(s), duration: %s, dt: %s, Iext: %s, Ibgk: %s, J: %s, tau: %s, W: %s, Gw: %s, initialrate: %s'%(Nareas, tstop, dt, Iext, Ibgk, J, tau, W, Gw, initialrate))
    rate = np.zeros((4, int(round(tstop/dt) + 1), Nareas))
    # Apply additional input current only on excitatory layers
    tstep2 = ((dt * sig * sig) / tau) ** .5
    mean_xi = 0
    std_xi = noise
    xi = np.random.normal(mean_xi, std_xi, (4, int(round(tstop/dt)) + 1, Nareas))
    
    # Initial rate values
    # Note: the 5 ensures that you have between 0 and 10 spikes/s
    rate[:, 0, :] = 5 * (1 + np.tanh(2 * xi[:, 0, :]))
    if initialrate>=0:
        rate[:, 0, :] = initialrate

    for dt_idx in range(len(t)):
        # iterate over different areas. Only true for the interareal simulation
        for area in range(Nareas):
            delta_rate = dt_calculate_rate(J, rate[:, dt_idx, area], Ibgk, Iext, dt, tau, tstep2, xi[:, dt_idx, area])
            rate[:, dt_idx + 1, area] = np.add(rate[:, dt_idx, area], delta_rate)

    # exclude the initial point that corresponds to the initial conditions
    rate = rate[:, 1:, :]

    return rate
