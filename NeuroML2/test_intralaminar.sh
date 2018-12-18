#!/bin/bash
set -e

#####   
##    This will test to make sure the intralaminar example matches between
##    Python and NeuroML implementations
##
##    See also https://github.com/OpenSourceBrain/MejiasEtAl2016/issues/3
#####

cd ../Python

# Run short duration test, no conns, no noise
python Mejias-2016.py -sigmaoverride 0 -analysis debug -noconns -testduration 1000 -initialrate 5

# Run short duration test, with conns, no noise
python Mejias-2016.py -sigmaoverride 0 -analysis debug -testduration 1000 -initialrate 5

# Run long duration test, with conns, with noise
python Mejias-2016.py -analysis debug -testduration 50000 -initialrate 1

# Run long duration test, no conns, with noise
python Mejias-2016.py -analysis debug -noconns -testduration 50000 -initialrate 1

cd ../NeuroML2

python GenerateNeuroMLlite.py -test
