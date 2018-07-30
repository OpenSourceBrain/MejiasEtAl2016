from __future__ import print_function, division

import numpy as np
import math
import matplotlib.pylab as plt


def transduction_function(x):
    # note: define boundary conditions for the transduction function
    if x == 0:
        return 1
    elif x <= -100:
        return 0
    else:
        return x / (1 - math.exp(-x))


def calculate_firing_rate(dt, re, ri, wee, wie, wei, wii, tau_e, tau_i, sei, xi_i, xi_e, Iext_e, Iext_i):
    tstep2e = ((dt * sei * sei) / tau_e) ** .5
    tstep2i = ((dt * sei * sei) / tau_i) ** .5
    dE = dt/tau_e * (-re + transduction_function((wee * re) + (wei * ri) + Iext_e)) + tstep2e * xi_e
    dI = dt/tau_i * (-ri + transduction_function((wie * re) + (wii * ri) + Iext_i)) + tstep2i * xi_i
    uu_p = re + dE
    vv_p = ri + dI
    return uu_p, vv_p


def calculate_rate(layer, t, dt, tstop, wee, wie, wei, wii, tau_e, tau_i, sei, Iext_e, Iext_i, noise, plot=False):
    uu_p = np.zeros((len(t) + 1, 1))
    vv_p = np.zeros((len(t) + 1, 1))

    mean_xi = 0
    std_xi = noise
    xi_e = np.random.normal(mean_xi, std_xi, int(round(tstop/dt)) + 1)
    xi_i = np.random.normal(mean_xi, std_xi, int(round(tstop/dt)) + 1)

    # Initial rate values
    # Note: the 5 ensures that you have between 0 and 10 spikes/s
    uu_p[0] = 5 * (1 + np.tanh(2 * xi_e[0]))
    vv_p[0] = 5 * (1 + np.tanh(2 * xi_i[0]))

    for dt_idx in range(len(t)):
        uu_p[dt_idx + 1], vv_p[dt_idx + 1] = calculate_firing_rate(dt, uu_p[dt_idx], vv_p[dt_idx], wee, wie, wei, wii,
                                                                   tau_e, tau_i, sei, xi_i[dt_idx], xi_e[dt_idx],
                                                                   Iext_e, Iext_i)

    if plot:
        tplot = np.linspace(0, tstop, tstop/dt + 1)
        fig = plt.figure()
        plt.plot(tplot, vv_p, label='vv', color='blue')
        plt.legend()
        plt.title(layer)
        plt.ylim(0.6, 0.8)
        plt.xlabel('Time')
        plt.ylabel('Proportion of firing cells')
        plt.savefig('E_activity.png')

        fig = plt.figure()
        plt.plot(tplot, uu_p, label='uu', color='red')
        plt.legend()
        plt.title(layer)
        plt.ylim(0.2, 0.6)
        plt.xlabel('Time')
        plt.ylabel('Proportion of firing cells')
        plt.savefig('I_activity.png')
        plt.show()

        fig = plt.figure()
        plt.plot(tplot, abs(uu_p - vv_p), label='sum', color='green')
        plt.title(layer)
        plt.ylabel('Proportion of firing cells')
        plt.xlabel('Time')
        plt.legend()
        plt.show()

    return uu_p, vv_p
