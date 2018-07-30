import os
import numpy as np
import matplotlib.pylab as plt

from calculate_rate import calculate_rate


def debug_neuroml(analysis, layer, t, dt, tstop, J, tau, sig, Iexts, nruns, noise, nogui):


    # calculate the firing rate
    for i in Iexts:
        # inject current only on excitatory layer
        Iext = np.array([[i], [0], [i], [0]])

        for nrun in nruns:
            rate = calculate_rate(t, dt, tstop, J, tau, sig, Iext, noise)

            # select only the excitatory and inhibitory layers for L2/3
            uu_p = np.expand_dims(rate[0, :], axis=1)
            vv_p = np.expand_dims(rate[1, :], axis=1)
            # Plot the layers time course
            plt.figure()
            plt.plot(uu_p, label='excitatory', color='r')
            plt.plot(vv_p, label='inhibitory', color='b')
            plt.ylim([-.5, 2])
            plt.legend()
            plt.title('noise=' + str(noise))

            # save the simulation as a txt file
            filename = os.path.join(analysis, layer + '_simulation_Iext_'+
                                    str(i) + '_nrun_' + str(nrun))
            activity = np.concatenate((uu_p, vv_p), axis=1)
            np.savetxt(filename + '.txt', activity)

    print('Done debugging!')
