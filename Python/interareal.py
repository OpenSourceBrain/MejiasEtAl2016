import numpy as np
from scipy.stats import ttest_ind


from calculate_rate import calculate_rate
from helper_functions import calculate_periodogram, find_peak_frequency


def interareal_simulation(t, dt, tstop, J, W, tau, Iext, Ibkg, sig, noise):
    # feedback selectiviyt
    Gw = 1

    Nareas = 2
    stat = 10 # TODO: Number of time to repeat the analysis
    powerpeak = np.zeros((4, stat))
    freqpeak = np.zeros((4, stat))
    nobs = t.shpae[0]
    binx = 10; eta=.2
    X = np.zeros((Nareas, int(round(nobs/binx)), stat))

    for jj in range(stat):
        calculate_rate(t, dt, tstop, J, tau, sig, Iext, Ibkg, noise, Nareas, W, Gw)
        # adapt the code so that it can take an array of Iext and return
        # mean input (if interareal_simulation). Interareal simulation also
        # takes an aditional argument W

    return fx2, px2, fx5, px5, powerpeak, fpeak, mean_input

def trialstat(rate, transient, dt, minfreq, nareas, stats):
    '''Main Interareal Analysis
        rate: Simulated rate
        transient: Number of points to averate over
        dt: dt of the simulation
        minfrequence: Frequencies below this threshold get discarded

    '''

    # To obtain some statistics salculate frequency and amplitude for multiple runs
    nlayers = 2 # we are only interested on the two excitatory layers

    powerpeak = np.zeros((nareas * nlayers, stats))
    freqpeak = np.zeros((nareas * nlayers, stats))
    # Calculate periodeogram to get the shape of the array
    pxx_t, fxx_t = calculate_periodogram(rate[0, :, 0], transient, dt)
    px2 = np.zeros((len(pxx_t), nareas, stats))
    fx2 = np.zeros((len(fxx_t), nareas, stats))
    px5 = np.zeros((len(pxx_t), nareas, stats))
    fx5 = np.zeros((len(fxx_t), nareas, stats))

    for stat in range(stats):
        k = 0
        for area in range(nareas):
            # Calculate power spectrum for the excitatory population only
            pxx2, fxx2 = calculate_periodogram(rate[area * 4, :, 0], transient, dt)

            # concatenate the results
            px2[:, area, stat] = pxx2
            fx2[:, area, stat] = fxx2

            frequency_l23, amplitudeA_l23, _, _ = \
                find_peak_frequency(fxx2, pxx2, minfreq, rate[0, :, 0])

            powerpeak[k, stat] = amplitudeA_l23
            freqpeak[k, stat] = frequency_l23

            k += 1

            pxx5, fxx5 = calculate_periodogram(rate[area * 4 + 3, :, 0], transient, dt)
            # concatenate the results
            px5[:, area, stat] = pxx5
            fx5[:, area, stat] = fxx5

            frequency_l56, amplitudeA_l56, _, _ = \
                find_peak_frequency(fxx5, pxx5, minfreq, rate[3, :, 0])

            powerpeak[k, stat] = amplitudeA_l56
            freqpeak[k, stat] = frequency_l56
            k += 1
    return fx2, px2, fx5, px5, powerpeak, freqpeak

def interareal_analysis(rate_rest, rate_stim, transient, dt, minfreq, nareas, stats):

    # Analysis of the simulation at rest
    fx20, px20, fx50, px50, powerpeak0, fpeak0 = trialstat(rate_rest, transient, dt, minfreq, nareas, stats)
    # Analysis of the simulation with additional stimulation
    fx2, px2, fx5, px5, powerpeak1, fpeak1 = trialstat(rate_stim, transient, dt, minfreq, nareas, stats)

    # Analysis after microstimulation
    # significance
    z1 = 3; z2 = 4; # Excitatory and inhibiory layers L5/6
    gamma0 = powerpeak0[z1, :]; gamma1 = powerpeak1[z1, :];
    alpha0 = powerpeak0[z2, :]; alpha1 = powerpeak1[z2, :];

    statistic, pvalue = ttest_ind(gamma0, gamma1)
    statistic2, pvalue2 = ttest_ind(alpha0, alpha1)




