#!/bin/bash
set -e

cd ../Python

# Run long duration test, with conns
python Mejias-2016.py -noise 1 -analysis debug -testduration 50000

# Run long duration test, no conns
#python Mejias-2016.py -noise 1 -analysis debug -noconns -testduration 50000

cd ../NeuroML2
