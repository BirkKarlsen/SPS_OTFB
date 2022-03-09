#!/bin/bash

# Make sure that the proper paths are used
source /afs/cern.ch/user/b/bkarlsen/.bashrc

# Run a simulation simulating four batches with impedances
python /afs/cern.ch/work/b/bkarlsen/Simulation_Files/SPS_OTFB/sim_files/OTFB_acq_simulation_ft_adi_real.py -nt 20000 -nr 10000 -oc 6 -vc 1 -fc 6 -sd gllrf0_nv_adi_fr4/
