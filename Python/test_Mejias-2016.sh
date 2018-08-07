#!/bin/bash

set -e

# Run one simulation for L2/3 layers when there is no additional external current
echo "Run simple simulation and save txt file"
python Mejias-2016.py -noise 1 -analysis debug_neuroML


