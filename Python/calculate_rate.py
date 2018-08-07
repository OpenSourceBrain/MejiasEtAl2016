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


def calculate_rate(t, dt, tstop, J, tau, sig, Iext, Ibgk, noise, Nareas):

    print('============================')
    print('  Calculating rates for <%s>'%(Nareas))
    print('    num t: %s'%(len(t)))
    print('    dt: %s'%(dt))
    print('    tstop: %s'%(tstop))
    print('    J: %s'%(J))
    print('----------------------------')
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
            total_input = Ibgk + Iext + np.expand_dims(np.dot(J, rate[:, dt_idx, area]), axis=1)
            # Do some transformations if analysis type is interareal:

            # calculate input after the transfer function
            transfer_input = transduction_function(total_input)
            delta_rate = dt / tau * (- np.expand_dims(rate[:, dt_idx, area], axis=1) + transfer_input) + \
                                     tstep2 * np.expand_dims(xi[:, dt_idx, area], axis=1)
            rate[:, dt_idx + 1, area] = np.squeeze(np.expand_dims(rate[:, dt_idx, area], axis=1) + delta_rate)

    # exclude the initial point that corresponds to the initial conditions
    rate = rate[:, 1:, :]
    
    print('  Calculated rates: %s'%len(rate))
    for i in [0,1,2,3]:
        print('    %i: %s -> %s'%(i,min(rate[i]),max(rate[i])))
        # print(len(rate[i]))
        # print(rate[i])
        u = np.expand_dims(rate[i, :], axis=1)
        # print('--')
        # print(len(u))
        # print(u)
    print('============================')

    return rate
