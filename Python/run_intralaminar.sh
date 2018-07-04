#!/bin/bash

set -e

python intralaminar.py -tau_e 6  -tau_i 15 -sei .3  -layer L2/3
python intralaminar.py -tau_e 30 -tau_i 75 -sei .45 -layer L5/6
