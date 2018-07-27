import pickle
import numpy as np
import matplotlib.pylab as plt
import sys


def plt_filled_std(ax, fxx_plt, data_mean, data_std, color, label):
    # calculate upper and lower bounds of the plot
    cis = (data_mean - data_std, data_mean + data_std)
    # plot filled area
    ax.fill_between(fxx_plt, cis[0], cis[1], alpha=0.2, color=color)
    # plot mean
    ax.plot(fxx_plt, data_mean, color=color, linewidth=2, label=label)
    ax.margins(x=0)

# load simulation results and plot
with open('intralaminar_simulation.pckl', 'rb') as filename:
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
if not '-nogui' in sys.argv:
    plt.show()

