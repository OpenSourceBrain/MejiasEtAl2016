import os
import numpy as np
import pickle
from scipy import signal
import matplotlib.pylab as plt
import math

from calculate_rate import calculate_rate

def down_sampled_periodogram(re, fft, fs, win, min_freq, transient, dt):

    # discard the first 25000 points
    restate = re[int(round((transient + dt)/dt)) - 1:]

    # perform periodogram on restate
    sq_data = np.squeeze(restate)
    fxx2, pxx2 = signal.periodogram(sq_data, fs=fs, window=win[25000:], nfft=fft, detrend=False, return_onesided=True,
                                    scaling='density')

    # Compress the data by sampling every 5 points.
    bin_size = 5
    # We start by calculating the number of index needed to in order to sample every 5 points and select only those
    remaining = fxx2.shape[0] - (fxx2.shape[0] % bin_size)
    fxx2 = fxx2[0:remaining]
    pxx2 = pxx2[0:remaining]
    # Then we calculate the average signal inside the specified non-overlapping windows of size bin-size.
    # Note: the output needs to be an np.array in order to be able to use np.where afterwards
    pxx_bin = np.asarray([np.mean(pxx2[i:i+bin_size]) for i in range(0, len(pxx2), bin_size)])
    fxx_bin = np.asarray([np.mean(fxx2[i:i+bin_size]) for i in range(0, len(fxx2), bin_size)])

    # find the frequency of the oscillations (note used at the moment)
    z = np.where(fxx_bin > min_freq)
    pxx_freq = pxx_bin[z]
    fxx_freq = fxx_bin[z]

    return pxx_bin, fxx_bin

def matlab_smooth(data, window_size):
    # asummes the data is one dimensional
    n = data.shape[0]
    c = signal.lfilter(np.ones(window_size)/window_size, 1, data)
    idx_begin = range(0, window_size - 2)
    cbegin = data[idx_begin].cumsum()
    # select every second elemeent and divide by their index
    cbegin = cbegin[0::2] / range(1, window_size - 1, 2)
    # select the list backwards
    idx_end = range(n-1, n-window_size + 1, -1)
    cend = data[idx_end].cumsum()
    # select every other element until the end backwards
    cend = cend[-1::-2] / (range(window_size - 2, 0, -2))
    c = np.concatenate([cbegin, c[window_size-1:], cend])
    return c

def intralaminar_analysis(Iexts, nruns, layer, dt, transient):
    # load pickle with all the results
    picklename = os.path.join('intralaminar', layer + '_simulation.pckl')
    with open(picklename, 'rb') as file1:
        simulation = pickle.load(file1)

    # sampling frequency to calculate the peridogram
    fs = 1/dt
    min_freq = 10
    psd_dic = {}

    for Iext in Iexts:
        psd_dic[Iext] = {}

        for nrun in range(nruns):

            psd_dic[Iext][nrun] = {}
            restate = simulation[Iext][nrun]['uu']
            N = restate.shape[0]
            win = signal.get_window('boxcar', N)
            # Calculate fft (number of freq. points at which the psd is estimated)
            # Calculate the max power of 2 and find the maximum value
            pow2 = int(round(math.log(N, 2)))
            fft = max(256, 2 ** pow2)

            # perform periodogram on restate.
            pxx_bin, fxx_bin = down_sampled_periodogram(restate, fft, fs, win,
                                                        min_freq, transient, dt)
            # smooth the data
            # Note: The matlab code transforms an even-window size into an odd number by subtracting by one.
            # So for simplicity I already define the window size as an odd number
            window_size = 79
            pxx = matlab_smooth(pxx_bin, window_size)

            psd_dic[Iext][nrun]['pxx'] = pxx
        # take the mean and std over the different runs
        psd_dic[Iext]['mean_pxx'] = np.mean([psd_dic[Iext][i]['pxx'] for i in range(nruns)], axis=0)
        psd_dic[Iext]['std_pxx'] = np.std([psd_dic[Iext][i]['pxx'] for i in range(nruns)], axis=0)

    # add fxx_bin to dictionary
    psd_dic['fxx_bin'] = fxx_bin

    for Iext in Iexts:
        fig = plt.figure()
        plt.semilogy(fxx_bin, psd_dic[Iext]['mean_pxx'])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (V**2/Hz)')
        plt.xlim(0, max(fxx_bin))

    # save the results into a pickle file
    picklename = os.path.join('intralaminar', layer + '_analysis.pckl')
    with open(picklename, 'wb') as file1:
        pickle.dump(psd_dic, file1)

    print('Done Analysis!')


def plt_filled_std(ax, fxx_plt, data_mean, data_std, color, label):
    # calculate upper and lower bounds of the plot
    cis = (data_mean - data_std, data_mean + data_std)
    # plot filled area
    ax.fill_between(fxx_plt, cis[0], cis[1], alpha=0.2, color=color)
    # plot mean
    ax.plot(fxx_plt, data_mean, color=color, linewidth=2, label=label)
    ax.margins(x=0)


def intralaminar_plt(layer):
    # load simulation results and plot
    analysis_pickle = os.path.join('intralaminar', layer + '_analysis.pckl')
    with open(analysis_pickle, 'rb') as filename:
        psd_dic = pickle.load(filename)

    # select only the first time points until fxx < 100
    fxx_plt_idx = np.where(psd_dic['fxx_bin'] < 100)
    fxx_plt = psd_dic['fxx_bin'][fxx_plt_idx]

    # find the correspondent mean and std pxx for this range
    Iexts = psd_dic.keys()
    # remove the fxx_bin key
    if 'fxx_bin' in Iexts:
        Iexts.remove('fxx_bin')
    for Iext in Iexts:
        psd_dic[Iext]['mean_pxx'] = psd_dic[Iext]['mean_pxx'][fxx_plt_idx]
        psd_dic[Iext]['std_pxx'] = psd_dic[Iext]['std_pxx'][fxx_plt_idx]

    # find the difference regarding the no_input
    psd_mean_0_0 = psd_dic[0]['mean_pxx'] - psd_dic[0]['mean_pxx']
    psd_mean_0_2 = psd_dic[2]['mean_pxx'] - psd_dic[0]['mean_pxx']
    psd_mean_0_4 = psd_dic[4]['mean_pxx'] - psd_dic[0]['mean_pxx']
    psd_mean_0_6 = psd_dic[6]['mean_pxx'] - psd_dic[0]['mean_pxx']

    # find the std
    psd_std_0_2 = np.sqrt(psd_dic[2]['std_pxx'] ** 2 + psd_dic[0]['std_pxx'] ** 2)
    psd_std_0_4 = np.sqrt(psd_dic[4]['std_pxx'] ** 2 + psd_dic[0]['std_pxx'] ** 2)
    psd_std_0_6 = np.sqrt(psd_dic[6]['std_pxx'] ** 2 + psd_dic[0]['std_pxx'] ** 2)

    lcolours = ['#588ef3', '#f35858', '#bd58f3']
    fig, ax = plt.subplots(1)
    plt_filled_std(ax, fxx_plt, psd_mean_0_2, psd_std_0_2, lcolours[0], 'Input = 2')
    plt_filled_std(ax, fxx_plt, psd_mean_0_4, psd_std_0_4, lcolours[1], 'Input = 4')
    plt_filled_std(ax, fxx_plt, psd_mean_0_6, psd_std_0_6, lcolours[2], 'Input = 6')
    plt.xlim([10, 80])
    plt.ylim([0, 0.003])
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Power (resp. rest)')
    plt.legend()


def intralaminar_simulation(analysis, layer, Iexts, nruns, t, dt, tstop,
                            wee, wei, wie, wii, tau_e, tau_i, sei, noise):
    simulation = {}
    for Iext in Iexts:

        simulation[Iext] = {}
        # run each combination of external input multiple times an take the average PSD

        for nrun in range(nruns):

            simulation[Iext][nrun] = {}
            # inject current only on excitatory layer
            Iext_e = Iext * 1
            Iext_i = 0
            uu_p, vv_p = calculate_rate(layer, t, dt, tstop, wee, wie, wei, wii, tau_e,
                                        tau_i, sei, Iext_e,
                                        Iext_i, noise, plot=False)

            simulation[Iext][nrun]['uu'] = uu_p
            simulation[Iext][nrun]['vv'] = vv_p

    picklename = os.path.join(analysis, layer + '_simulation.pckl')
    with open(picklename, 'wb') as file1:
        pickle.dump(simulation, file1)
    print('Done Simulation!')
