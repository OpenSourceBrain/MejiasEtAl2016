#!/bin/bash
set -e

cd ../Python

python Mejias-2016.py -analysis debug -noconns -testduration 50000 -dt 2e-4 -initialrate 1

python Mejias-2016.py -analysis debug -noconns -testduration 50000 -dt 2e-5 -initialrate 1


cd ../NeuroML2
