import os
import numpy as np
import matplotlib.pylab as plt

from calculate_rate import calculate_rate


def debug_neuroml(analysis, layer, t, dt, tstop, wee, wei, wie, wii, tau_e,
                  tau_i, sei, Iexts, nruns, noise, nogui):


    # calculate the firing rate
    for Iext in Iexts:
        # inject current only on excitatory layer
        Iext_e = Iext * 1
        Iext_i = 0

        for nrun in nruns:
            uu_p, vv_p = calculate_rate(layer, t, dt, tstop, wee, wie, wei, wii,
                                       tau_e, tau_i, sei, Iext_e, Iext_i,
                                       noise)

            # Plot the layers time course
            plt.figure()
            plt.plot(uu_p, label='excitatory', color='r')
            plt.plot(vv_p, label='inhibitory', color='b')
            plt.ylim([-.5, 2])
            plt.legend()
            plt.title('noise=' + str(noise))

            # save the simulation as a txt file
            filename = os.path.join(analysis, layer + '_simulation_Iext_'+
                                    str(Iext) + '_nrun_' + str(nrun))
            activity = np.concatenate((uu_p, vv_p), axis=1)
            np.savetxt(filename + '.txt', activity)

            if not nogui:
                plt.show()
    print('Done debugging!')
