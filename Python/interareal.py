import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pylab as plt

from calculate_rate import calculate_rate
from helper_functions import calculate_periodogram, find_peak_frequency, matlab_smooth, plt_filled_std


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

def trialstat(rate, transient, dt, minfreq_l23, minfreq_l56, nareas, stats):
    '''Main Interareal Analysis
        rate: Simulated rate
        transient: Number of points to averate over
        dt: dt of the simulation
        minfrequence: Frequencies below this threshold get discarded

    '''

    # To obtain some statistics calculate frequency and amplitude for multiple runs

    powerpeak = np.zeros((nareas * nareas, stats))
    freqpeak = np.zeros((nareas * nareas, stats))
    # Calculate periodeogram to get the shape of the array
    pxx_t, fxx_t = calculate_periodogram(rate[0, :, 0, 0], transient, dt)
    px2 = np.zeros((len(pxx_t), nareas, stats))
    fx2 = np.zeros((len(fxx_t), nareas, stats))
    px5 = np.zeros((len(pxx_t), nareas, stats))
    fx5 = np.zeros((len(fxx_t), nareas, stats))

    for stat in range(stats):
        k = 0
        for area in range(nareas):
            # Calculate power spectrum for the excitatory population from layer L23
            pxx2, fxx2 = calculate_periodogram(rate[area * 4, :, 0, stat], transient, dt)

            # concatenate the results
            px2[:, area, stat] = pxx2
            fx2[:, area, stat] = fxx2

            frequency_l23, amplitudeA_l23, _, _ = \
                find_peak_frequency(fxx2, pxx2, minfreq_l23, rate[area * 4, :, 0, stat])

            powerpeak[k, stat] = amplitudeA_l23
            freqpeak[k, stat] = frequency_l23

            k += 1

            # Calculate power spectrum for the excitatory population from layer L56
            pxx5, fxx5 = calculate_periodogram(rate[area * 4 + 2, :, 0, stat], transient, dt)
            # concatenate the results
            px5[:, area, stat] = pxx5
            fx5[:, area, stat] = fxx5

            frequency_l56, amplitudeA_l56, _, _ = \
                find_peak_frequency(fxx5, pxx5, minfreq_l56, rate[area * 4 + 2, :, 0, stat])

            powerpeak[k, stat] = amplitudeA_l56
            freqpeak[k, stat] = frequency_l56
            k += 1
    return fx2, px2, fx5, px5, powerpeak, freqpeak

def interareal_analysis(rate_rest, rate_stim, transient, dt, minfreq_l23, minfreq_l56, nareas, stats):

    # Analysis of the simulation at rest
    fx20, px20, fx50, px50, powerpeak0, fpeak0 = trialstat(rate_rest, transient, dt, minfreq_l23, minfreq_l56, nareas, stats)
    # Analysis of the simulation with additional stimulation
    fx2, px2, fx5, px5, powerpeak1, fpeak1 = trialstat(rate_stim, transient, dt, minfreq_l23, minfreq_l56, nareas, stats)

    # Analysis after microstimulation
    # significance
    #  Note: We are using -1 because Python starts with 0 indexing
    z1 = 3-1; z2 = 4-1; # Excitatory and inhibitory layers L5/6 (feedforward)
    gamma0 = powerpeak0[z1, :]; gamma1 = powerpeak1[z1, :]
    alpha0 = powerpeak0[z2, :]; alpha1 = powerpeak1[z2, :]

    statistic, pgamma = ttest_ind(gamma0, gamma1)
    statistic2, palpha = ttest_ind(alpha0, alpha1)

    return px20, px2, fx2

def interareal_plt(area, px20, px2, fx2):
    '''

    area: 0 for V1 and 2 for V2
    '''

    barrasgamma = np.zeros((2,2))
    barrasalpha = np.zeros((2,2))


    ## Analysis for rest model
    # reshuffle the px20 data so that you have stats x time points for the area of interst
    pz0 = np.squeeze(np.transpose(px20[:, area, :]))
    fx0 = np.transpose(fx2[:, area, 0])
    # smooth the data
    # Note: The matlab code transforms an even-window size into an odd number by subtracting by one.
    # So for simplicity I already define the window size as an odd number
    window_size = 99
    pxx0 = []
    lcolours = ['#1F5E43', '#31C522', '#2242C5']
    for i in range(len(pz0)):
        pxx0.append(matlab_smooth(pz0[i, :], window_size))
    pxx0 = np.asarray(pxx0)
    pxx20 = np.mean(pxx0, axis=0)
    pxx20sig = np.std(pxx0, axis=0)

    fxx_plt_idx = np.where((fx0 > 20) & (fx0 < 80))
    z1 = pxx20[fxx_plt_idx]
    z2 = pxx20sig[fxx_plt_idx]
    b2 = np.argmax(z1)
    barrasgamma[0, 0] = z1[b2]
    barrasgamma[0, 1] = z2[b2]

    ## Analysis for model with stimulus
    pz = np.squeeze(np.transpose(px2[:, area, :]))
    fx = np.transpose(fx2[:, area, 1])
    pxx = []
    for i in range(len(pz)):
        pxx.append(matlab_smooth(pz[i, :], window_size))
    pxx = np.asarray(pxx)
    pxx2 = np.mean(pxx, axis=0)
    pxx2sig = np.std(pxx, axis=0)
    fxx_plt_idx = np.where((fx > 20) & (fx < 80))
    z1 = pxx2[fxx_plt_idx]
    z2 = pxx2sig[fxx_plt_idx]
    b2 = np.argmax(z1)
    barrasgamma[1, 0] = z1[b2]
    barrasgamma[1, 1] = z2[b2]

    # Plot results for V4 L2/3
    fig, ax = plt.subplots(1)
    resbin2 = 20
    plt_filled_std(ax, fx0[1:-1:resbin2], pxx20[1:-1:resbin2], pxx20sig[1:-1:resbin2], lcolours[0], 'rest')
    plt_filled_std(ax, fx[1:-1:resbin2], pxx2[1:-1:resbin2], pxx2sig[1:-1:resbin2], lcolours[1], 'stimulus')
    plt.xlim([20, 80])
    plt.ylim([0, .006])
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Power (resp. rest)')
    plt.legend()
    plt.savefig('interareal/V4_l23.png')

    filter = [20, 80]
    def plot_plot(filter):


        return fx, pxx, pxx0, pxxsig, pxx0sig, z1, z2

    # Plot results for V4 L5/6
    # fig, ax = plt.subplots(1)
    # resbin5 = 10
    # plt_filled_std(ax, fx[1:-1:resbin5], pxx20[1:-1:resbin5], pxx20sig[1:-1:resbin5], lcolours[0], 'rest')
    # plt_filled_std(ax, fx[1:-1:resbin5], pxx2[1:-1:resbin5], pxx2sig[1:-1:resbin5], lcolours[1], 'stimulus')
    # plt.xlim([5, 20])
    # plt.ylim([0, .025])
    # plt.xlabel('Frequency(Hz)')
    # plt.ylabel('Power (resp. rest)')
    # plt.legend()
    # plt.show()


