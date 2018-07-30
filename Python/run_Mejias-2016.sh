#!/bin/bash

set -e

# Run one simulation for L2/3 layers when there is no additional external current
echo "Run simple simulation and save txt file"
python Mejias-2016.py -noise 1 -analysis debug_neuroML

# Note: the values for tau in the paper represent the values in ms. Here we use
# s.
# This will create the Fig2b from the Mejias-2016 paper
echo "Run intralaminar simulation for L2/3...(it will take a few mins)"
python Mejias-2016.py -noise 1 -analysis intralaminar

