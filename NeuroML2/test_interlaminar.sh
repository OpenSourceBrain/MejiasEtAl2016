#!/bin/bash
set -e

#####   
##    This will test to make sure the interlaminar example matches between
##    Python and NeuroML implementations
##
##    See also https://github.com/OpenSourceBrain/MejiasEtAl2016/pull/13
#####

cd ../Python

if [ -f interlaminar_a/simulation.pckl ]; then
    echo "Remove previous simulation file"
    rm interlaminar_a/simulation.pckl
fi
python Mejias-2016.py -analysis interlaminar_a -nogui


cd ../NeuroML2

python GenerateNeuroMLlite.py -interlaminar -jnml
