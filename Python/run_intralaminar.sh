#!/bin/bash

set -e

# Note: the values for tau in the paper represent the values in ms. Here we use
# s.
echo "Runing intralaminar simulation for L2/3...(it will take a few min)"
python intralaminar_simulation.py -tau_e .006  -tau_i .015 -sei .3  -layer L2/3

# plot Figure 2b
# Note: this script assumes that the simulation has been saved a dictionary
python intralaminar_plt.py

# Todo: For now the simulation is run only for L2/3
# python intralaminar_simulation.py -tau_e .030 -tau_i .075 -sei .45 -layer L5/6
