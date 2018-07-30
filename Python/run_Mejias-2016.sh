#!/bin/bash

set -e

# Run one simulation for L2/3 layers when there is no additional external current
echo "Run simple simulation and save txt file"
python Mejias-2016.py -tau_e .006 -tau_i .015 -layer L2_3 -sei .3 -noise 1 \
     -analysis debug_neuroML

# Note: the values for tau in the paper represent the values in ms. Here we use
# s.
# This will create the Fig2b from the Mejias-2016 paper
echo "Run intralaminar simulation for L2/3...(it will take a few min)"
python Mejias-2016.py -tau_e .006 -tau_i .015 -layer L2_3 -sei .3 -noise 1\
    -analysis intralaminar

# Todo: For now the simulation is run only for L2/3
# python intralaminar_simulation.py -tau_e .030 -tau_i .075 -sei .45 -layer L5/6
