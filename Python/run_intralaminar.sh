#!/bin/bash

set -e

# Note: the values for tau in the paper represent the values in ms. Here we use
# s.
python intralaminar.py -tau_e .006  -tau_i .015 -sei .3  -layer L2/3
python intralaminar.py -tau_e .030 -tau_i .075 -sei .45 -layer L5/6

# plot Figure 2b
# Note: this script assumes that the simulation has been saved a dictionary
python intralaminar_plt.py
