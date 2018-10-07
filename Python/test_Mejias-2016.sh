#!/bin/bash

set -e

# Run one simulation for L2/3 layers when there is no additional external current
echo "Run simple simulation and save txt file"
python Mejias-2016.py -noise 1 -analysis debug

# Note: the values for tau in the paper represent the values in ms. Here we use
# s.
# This will create the Fig2b from the Mejias-2016 paper
echo "------------------------------------------------------------------------"
echo "Run intralaminar simulation for L2/3"
if [ -f intralaminar/L2_3_simulation.pckl ]; then
    echo "Remove previous simulation file"
    rm intralaminar/L2_3_simulation.pckl
fi
python Mejias-2016.py -noise 1 -analysis intralaminar -nogui

echo "------------------------------------------------------------------------"
echo "Generate Power Spectrum for L2/3 and L5/6"
if [ -f interlaminar_a/simulation.pckl ]; then
    echo "Remove previous simulation file"
    rm interlaminar_a/simulation.pckl
fi
python Mejias-2016.py -noise 1 -analysis interlaminar_a -nogui

echo "------------------------------------------------------------------------"
echo "Generate a set of 30 traces of activity in layer 5/6 (in gray) and their
average (in blue) L5/6"
if [ -f interlaminar_b/simulation.pckl ]; then
    echo "Remove previous simulation file"
    rm interlaminar_b/simulation.pckl
fi
python Mejias-2016.py -noise 1 -analysis interlaminar_b -nogui
