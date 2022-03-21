'''
File to launch multiple simulations to scan values of the LLRF gain effect.

Author: Birk Emil Karlsen-Bæck
'''

# Import ----------------------------------------------------------------------
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="This file launches simulations for parameter scans.")

parser.add_argument("--freq_config", "-fc", type=int,
                    help="Different configurations of the TWC central frequencies.")

args = parser.parse_args()

# Parameter Values ------------------------------------------------------------
input_array = np.array([1, 2, 3, 4, 5])
gllrf_array = np.array([5, 10, 14, 16, 20])
FREQ_CONFIG = 3
bash_dir = '/afs/cern.ch/work/b/bkarlsen/Simulation_Files/SPS_OTFB/bash_files/'
sub_dir = '/afs/cern.ch/work/b/bkarlsen/Submittion_Files/SPS_OTFB/'

if args.freq_config is not None:
    FREQ_CONFIG = args.freq_config


# Make necessary preparations for Sims ----------------------------------------

bash_file_names = np.zeros(input_array.shape).tolist()
sub_file_names = np.zeros(input_array.shape).tolist()
file_names = np.zeros(input_array.shape).tolist()

print('\nMaking shell scripts...')

for i in range(len(input_array)):
    bash_file_names[i] = f'scan_fr{FREQ_CONFIG}_llrf_{gllrf_array[i]:.0f}.sh'
    sub_file_names[i] = f'scan_fr{FREQ_CONFIG}_llrf_{gllrf_array[i]:.0f}.sub'
    file_names[i] = f'scan_fr{FREQ_CONFIG}_llrf_{gllrf_array[i]:.0f}'

    # Make bash file
    os.system(f'touch {bash_dir}{bash_file_names[i]}')

    bash_content = f'#!/bin/bash\n' \
                   f'source /afs/cern.ch/user/b/bkarlsen/.bashrc\n' \
                   f'python /afs/cern.ch/work/b/bkarlsen/Simulation_Files/SPS_OTFB/sim_files/' \
                   f'OTFB_acq_simulation_ft_real.py -nt 30000 -nr 0 -oc 1 -vc 1 -fc {FREQ_CONFIG} ' \
                   f'-gc {input_array[i]} -sd scan_fr{FREQ_CONFIG}_llrf_{gllrf_array[i]:.0f}/'

    os.system(f'echo "{bash_content}" > {bash_dir}{bash_file_names[i]}')
    os.system(f'chmod a+x {bash_dir}{bash_file_names[i]}')


print('\nMaking and submitting simulations...')
for i in range(len(input_array)):
    # Make submission file
    os.system(f'touch {sub_dir}{sub_file_names[i]}')

    sub_content = f'executable = {bash_dir}{bash_file_names[i]}\n' \
                  f'arguments = \$(ClusterId)\$(ProcId)\n' \
                  f'output = {bash_dir}{file_names[i]}.\$(ClusterId)\$(ProcId).out\n' \
                  f'error = {bash_dir}{file_names[i]}.\$(ClusterId)\$(ProcId).err\n' \
                  f'log = {bash_dir}{file_names[i]}.\$(ClusterId)\$(ProcId).log\n' \
                  f'+JobFlavour = \\"tomorrow\\"\n' \
                  f'queue'

    os.system(f'echo "{sub_content}" > {sub_dir}{sub_file_names[i]}')
    os.system(f'chmod a+x {sub_dir}{sub_file_names[i]}')

    os.system(f'condor_submit {sub_dir}{sub_file_names[i]}')
