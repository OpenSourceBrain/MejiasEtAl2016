#!/bin/bash

set -e

#####   
##    This will test to make sure the NeuroML model gives similar results 
##    independent of the timestep, dt
##
##    See also https://github.com/OpenSourceBrain/MejiasEtAl2016/issues/3
#####

cd ../Python

python Mejias-2016.py -analysis debug -noconns -testduration 50000 -dt 2e-4 -initialrate 1

python Mejias-2016.py -analysis debug -noconns -testduration 50000 -dt 2e-5 -initialrate 1

cd ../NeuroML2

python GenerateNeuroMLlite.py -dt
