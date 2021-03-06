'''
Code that reads the csv files downloaded from the Kennedy lab and save them in
the correct way to be analysed by the Mejias2016 code
'''

import pandas as pd
import numpy as np
from scipy.io import savemat


def sort_list(list1, list2):
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)]
    return z

# FLN, SLN and Areas
m1 = pd.read_csv('Neuron_2015_Table.csv')
# rename the targets and source from '8B' to '8b'
m1['TARGET'] = m1['TARGET'].replace(['8B'],'8b')
m1['SOURCE'] = m1['SOURCE'].replace(['8B'],'8b')
# Get the list of regions
targets = np.unique(m1['TARGET'])
# sort the list of targets according to the region's rank
targets_ranked = np.array(['V1', 'V2', 'V4', 'DP', 'MT', '8m', '5', '8l', '2',
                           'TEO', 'F1', 'STPc', '7A', '46d', '10', '9/46v',
                           '9/46d', 'F5', 'TEpd', 'PBr', '7m', 'F2', '7B',
                           'ProM', 'STPi', 'F7', '8b', 'STPr', '24c'])
# check if there is any region on the ranked list that is missing
if not len(set(targets_ranked) - set(targets)) == 0:
    ValueError("Both Sets have a different number of brain regions")
# find index of the regions assorted
indx = [np.where(targets_ranked==target)[0][0] for target in targets]

# For all areas present in the target list find the corresponding FLN and SLN
# and save them in the correct format (areas x areas)
# NOTE: the data for the LIP region is not present
flnMat = np.zeros((len(targets), len(targets)))
slnMat = np.zeros((len(targets), len(targets)))

mytargets = []
mysources = []
for i in range(len(m1['TARGET'])):
    if m1['TARGET'][i] in targets_ranked and m1['SOURCE'][i] in targets_ranked:
        print('Target: %s; Source %s') %(str(m1['TARGET'][i]),
                                         str(m1['SOURCE'][i]) )
        target_idx = np.where(targets_ranked == m1['TARGET'][i])
        source_idx = np.where(targets_ranked == m1['SOURCE'][i])
        flnMat[target_idx, source_idx] = m1['FLN'][i]
        slnMat[target_idx, source_idx] = m1['SLN'][i]

matdic = {}
# sort fln and sln according to indx
#print(sort_list(indx, targets))
matdic['flnMat'] = sort_list(flnMat, indx)
matdic['slnMat'] = sort_list(slnMat, indx)
matdic['areaList'] = sort_list(targets, indx)
if not (matdic['areaList'] == targets_ranked).all:
    ValueError('The two sorted lists are not identical')

# save the results in a mat file
savemat('../Matlab/fig6/subgraphData30.mat', matdic)

#### Wiring
wiring_file = pd.read_excel('PNAS_2013_Distance_Matrix.xlsx')
# select only the regions in the target list (which corresponds to the columns
# wiring variable)

# convert all indexes and columns into strings
wiring_file.index = wiring_file.index.map(str)
wiring_file.columns = wiring_file.columns.map(str)
# rename the index '8B' to '8b' to keep consistency between sources and targets
wiring_file.rename(index={'8B':'8b'}, inplace=True)
# rename the indexes that are different between the two excel files
wiring_file.rename(index={'Tepd': 'TEpd'}, inplace=True)
wiring_file.rename(columns={'Tepd': 'TEpd'}, inplace=True)

# the entry wiring['9/46v']['9/46v'] is empty convert it to NaN
wiring_file['9/46v']['9/46v'] = np.nan

wiring_mat = np.zeros((len(targets), len(targets)))
for idx1, region1 in enumerate(targets_ranked):
    for idx2, region2 in enumerate(targets_ranked):
        wiring_mat[idx1, idx2] = wiring_file[region1][region2]

wiring_dic = {}
wiring_dic['wiring'] = wiring_mat

savemat('../Matlab/fig6/subgraphWiring30.mat', wiring_dic)

# plot the 8 regions of interest
print('Selected ROIs')
idx = [0, 1, 2, 3, 5, 6, 8, 12]
print(targets_ranked[idx])
