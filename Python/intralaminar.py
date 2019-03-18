import os
import numpy as np
import pickle
from scipy import signal
import matplotlib.pylab as plt

from calculate_rate import calculate_rate
from helper_functions import calculate_periodogram, compress_data, plt_filled_std, matlab_smooth


def intralaminar_analysis(simulation, Iexts, nruns, layer='L23', dt=2e-04, transient=5):
    """
    Calculates the main intralaminar analysis and dumps a pickle containing the periodogram of the analysis
    Inputs
        simulation: dictionary containing all the simulations to be analysed
        Iexts: a list of the input strengths applied on the excitatory populations
        nruns: number of simulations analysed for every Iext
        layer: Layer under analysis
        dt: time step of the simulation
        transient:

    """

    psd_dic = {}

    for Iext in Iexts:
        psd_dic[Iext] = {}

        for nrun in range(nruns):

            psd_dic[Iext][nrun] = {}
            restate = simulation[Iext][nrun]['L23_E/0/L23_E/r']

            # perform periodogram on restate.
            pxx2, fxx2 = calculate_periodogram(restate, transient, dt)

            # Compress the data by sampling every 5 points.
            bin_size = 5
            pxx_bin, fxx_bin = compress_data(pxx2, fxx2, bin_size)

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

    print('    Done Analysis!')
    return psd_dic



def intralaminar_plt(psd_dic):
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
    if not os.path.exists('intralaminar'):
        os.makedirs('intralaminar')
    plt.savefig('intralaminar/intralaminar.png')


def intralaminar_simulation(analysis, layer, Iexts, Ibgk, nruns, t, dt, tstop,
                            J, tau, sig, noise, Nareas):
    simulation = {}
    for Iext in Iexts:
        simulation[Iext] = {}
        Iext_a = np.array([Iext, 0, Iext, 0])
        # run each combination of external input multiple times an take the average PSD
        for nrun in range(nruns):

            simulation[Iext][nrun] = {}
            rate = calculate_rate(t, dt, tstop, J, tau, sig, Iext_a, Ibgk, noise, Nareas)

            # Note: Save only the excitatory and inhibitory signal from L2/3.
            # For compatibility with NeuroML/LEMS transform the results into a row matrix
            simulation[Iext][nrun]['L23_E/0/L23_E/r'] = rate[0, :].reshape(-1)
            simulation[Iext][nrun]['L23_I/0/L23_I/r'] = rate[1, :].reshape(-1)

    picklename = os.path.join(analysis, layer + '_simulation.pckl')
    with open(picklename, 'wb') as file1:
        pickle.dump(simulation, file1)
    print('    Done Simulation!')
    return simulation
