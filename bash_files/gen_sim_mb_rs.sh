#!/bin/bash

# Make sure that the proper paths are used
source /afs/cern.ch/user/b/bkarlsen/.bashrc

# Run a simulation simulating four batches with impedances
python /afs/cern.ch/work/b/bkarlsen/Simulation_Files/SPS_OTFB/sim_files/OTFB_acq_simulation_ft_gen.py -cf 3 -nt 10000 -sb 0 -g 0 -md reversed_sign/ -ns 1000 -np 1000 -ri -1
