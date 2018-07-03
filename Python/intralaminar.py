import numpy as np
import math
import matplotlib.pylab as plt
import argparse

def transduction_function(x):
    # note: define boundary conditions for the transduction function
    if x == 0:
        return 1
    elif x <= -100:
        return 0
    else:
        return x / (1 - math.exp(-x))


def calculate_firing_rate(dt, re, ri, wee, wie, wei, wii, tau_e, tau_i, sei):
    dE = dt * (-re + transduction_function((wee * re) + (wei * ri)) + math.sqrt(tau_e) * np.random.normal(0, sei))/tau_e
    dI = dt * (-ri + transduction_function((wie * re) + (wii * ri)) + math.sqrt(tau_i) * np.random.normal(0, sei))/tau_i
    uu_p = re + dE
    vv_p = ri + dI
    return uu_p, vv_p

parser = argparse.ArgumentParser(description='Parameters for the simulation')
parser.add_argument('-tau_e', type=float, dest='tau_e', help='Excitatory membrane time constant (tau_e)')
parser.add_argument('-tau_i', type=float, dest='tau_i', help='Inhibitory membrane time constant (tau_i)')
parser.add_argument('-sei', type=float, dest='sei', help='Deviation for the Gaussian white noise (s_ei)')
parser.add_argument('-layer', type=str, dest='layer', help='Layer of interest')

args = parser.parse_args()

## superficial layer
wee = 1.5
wei = -3.25
wie = 3.5
wii = -2.5

dt = .05
tstop = 700
t = np.linspace(0, tstop, tstop/dt)
uu_p = np.zeros((len(t) + 1, 1))
vv_p = np.zeros((len(t) + 1, 1))
# note: I am assuming the initial condition is close to zero (not specified on the paper)
uu_p[0] = 0.0001
vv_p[0] = 0.0001

for dt_idx in range(len(t)):
    uu_p[dt_idx + 1], vv_p[dt_idx + 1] = calculate_firing_rate(dt, uu_p[dt_idx], vv_p[dt_idx], wee, wie, wei, wii,
                                                               args.tau_e, args.tau_i, args.sei)

tplot = np.linspace(0, tstop, tstop/dt + 1)
plt.plot(tplot, uu_p, label='uu', color='red')
plt.plot(tplot, vv_p, label='vv', color='blue')
plt.legend()
plt.xlim([350, 700])
plt.title(args.layer)
plt.xlabel('Time')
plt.ylabel('Proportion of firing cells')
plt.savefig('EI_activity.png')

plt.figure()
plt.plot(tplot, abs(uu_p - vv_p), label='sum', color='green')
plt.xlim([350, 700])
plt.title(args.layer)
plt.ylabel('Proportion of firing cells')
plt.xlabel('Time')
plt.legend()
plt.show()
