import os
import numpy as np
import matplotlib.pylab as plt
from scipy import signal
import math

from calculate_rate import calculate_rate


def debug_neuroml(analysis, layer, t, dt, tstop, J, tau, sig, Iexts, Ibgk, nruns, noise, nogui, Nareas):
    # calculate the firing rate
    for i in Iexts:
        # inject current only on excitatory layer
        Iext = np.array([[i], [0], [i], [0]])

        for nrun in nruns:
            rate = calculate_rate(t, dt, tstop, J, tau, sig, Iext, Ibgk, noise, Nareas)
            
            # select only the excitatory and inhibitory layers for L2/3
            uu_p = np.expand_dims(rate[0, :, 0], axis=1)
            vv_p = np.expand_dims(rate[1, :, 0], axis=1)
            
            transient = .2
            # perform periodogram on restate.
            pxx2_u, fxx2_u = calculate_periodogram(uu_p, transient, dt)
            pxx2_v, fxx2_v = calculate_periodogram(vv_p, transient, dt)
            # print('pxx2: %s (%s); fxx2: %s (%s)'%(pxx2, len(pxx2), fxx2, len(fxx2)))

            plt.figure()
            plt.plot(fxx2_u, pxx2_u, color='r')
            plt.plot(fxx2_v, pxx2_v, color='b')
            plt.xlim([-2, 80])
            
            # Plot the layers time course
            plt.figure()
            plt.plot(t,uu_p, label='excitatory', color='r')
            plt.plot(t,vv_p, label='inhibitory', color='b')
            plt.ylim([-.5, 2])
            plt.legend()
            plt.title('noise=' + str(noise))

            # save the simulation as a txt file
            filename = os.path.join(analysis, layer + '_simulation_Iext_'+
                                    str(i) + '_nrun_' + str(nrun))
          
            activity = np.concatenate(([[tt] for tt in t], uu_p, vv_p), axis=1)
            
            np.savetxt(filename + '.txt', activity)

    print('Done debugging!')


def calculate_periodogram(re, transient, dt):
    # calculate fft and sampling frequency for the peridogram
    fs = 1 / dt
    N = re.shape[0]
    win = signal.get_window('boxcar', N)
    # Calculate fft (number of freq. points at which the psd is estimated)
    # Calculate the max power of 2 and find the maximum value
    pow2 = int(round(math.log(N, 2)))
    fft = max(256, 2 ** pow2)

    # discard the first points
    restate = re[int(round((transient + dt)/dt)) - 1:]

    # perform periodogram on restate
    sq_data = np.squeeze(restate)
    fxx2, pxx2 = signal.periodogram(sq_data, fs=fs, window=win[int(round((transient + dt)/dt)) - 1:],
                                    nfft=fft, detrend=False, return_onesided=True,
                                    scaling='density')
    return pxx2, fxx2

if __name__ == '__main__':
    
    import random
    dt = 0.001 
    tmax = 2
    times = np.array([i*dt for i in range(int(tmax/dt)+1)])
    
    noise = 0.5
    period1 = 0.03
    
    a = np.array([ math.sin(t/period1)+noise*random.random() for t in times])
    print len(times)
    print a
    
    plt.figure()
    plt.plot(times,a, color='r')
    plt.legend()
    plt.title('noise = %s; dt = %s'%(noise,dt))
    
    transient = 1
    print a
    aa = np.array([[a]])
    print aa
    uu = np.expand_dims(aa, axis=1)
    print uu
    pxx2, fxx2 = calculate_periodogram(uu,transient, dt)
    print pxx2
    print fxx2
    
    plt.show()
    