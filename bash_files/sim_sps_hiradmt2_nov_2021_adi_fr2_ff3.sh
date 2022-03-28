#!/bin/bash

# Make sure that the proper paths are used
source /afs/cern.ch/user/b/bkarlsen/.bashrc

# Run a simulation simulating four batches with impedances
python /afs/cern.ch/work/b/bkarlsen/Simulation_Files/SPS_OTFB/sim_files/OTFB_acq_simulation_ft_real.py -nt 30000 -nr 0 -oc 3 -vc 1 -fc 2 -ff 1 -sd gllrf16_nv_fr2_ff3/ -fir 3
