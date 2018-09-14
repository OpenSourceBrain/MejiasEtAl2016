#!/bin/bash
set -e

cd ../Python

# Run short duration test, no conns
python Mejias-2016.py -noise 0 -analysis debug -noconns -testduration 1000 -initialrate 5

# Run short duration test, with conns
python Mejias-2016.py -noise 0 -analysis debug -testduration 1000 -initialrate 5

# Run long duration test, with conns
python Mejias-2016.py -noise 1 -analysis debug -testduration 50000 -initialrate 5

# Run long duration test, no conns
python Mejias-2016.py -noise 1 -analysis debug -noconns -testduration 50000 -initialrate 5

cd ../NeuroML2
