'''
File to launch multiple simulations to scan values of the transmitter gain effect.

Author: Birk Emil Karlsen-Bæck
'''

# Import ----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import argparse
import os


# Parameter Values ------------------------------------------------------------
ratio_array = np.linspace(0.90, 1.10, 10)
ratio_array_extended = np.linspace(0.80, 1.20, 10)
FREQ_CONFIG = 1
bash_dir = '/afs/cern.ch/work/b/bkarlsen/Simulation_Files/SPS_OTFB/bash_files/'
sub_dir = '/afs/cern.ch/work/b/bkarlsen/Submittion_Files/SPS_OTFB/'


# Make necessary preparations for Sims ----------------------------------------

# TODO: Make bash and submittion files for each simulation

bash_file_names = np.zeros(ratio_array.shape, dtype=str)
sub_file_names = np.zeros(ratio_array.shape, dtype=str)
file_names = np.zeros(ratio_array.shape, dtype=str)

for i in range(len(ratio_array)):
    bash_file_names[i] = f'scan_fr{FREQ_CONFIG}_tx_{100 * ratio_array[i]:.0f}.sh'
    sub_file_names[i] = f'scan_fr{FREQ_CONFIG}_tx_{100 * ratio_array[i]:.0f}.sub'
    file_names[i] = f'scan_fr{FREQ_CONFIG}_tx_{100 * ratio_array[i]:.0f}'

    # Make bash file
    os.system(f'touch {bash_dir}{bash_file_names[i]}')

    # TODO: Make parser for transmitter gain ratios
    bash_content = f'#!/bin/bash\n' \
                   f'source /afs/cern.ch/user/b/bkarlsen/.bashrc\n' \
                   f'python /afs/cern.ch/work/b/bkarlsen/Simulation_Files/SPS_OTFB/sim_files/' \
                   f'OTFB_acq_simulation_ft_adi_real.py -nt 20000 -nr 1 -oc 3 -vc 1 -fc {FREQ_CONFIG} ' \
                   f'-sd scan_fr{FREQ_CONFIG}_tr_{ratio_array[i]}/'

    os.system(f'echo "{bash_content}" > {bash_dir}{bash_file_names[i]}')
    os.system(f'chmod a+x {bash_dir}{bash_file_names[i]}')

for i in range(len(ratio_array)):
    # Make submittion file
    os.system(f'touch {sub_dir}{sub_file_names[i]}')

    sub_content = f'executable = {bash_dir}{bash_file_names[i]}\n' \
                  f'arguments = $(ClusterId)$(ProcId)\n' \
                  f'output = {bash_dir}{file_names[i]}.$(ClusterId)$(ProcId).out\n' \
                  f'error = {bash_dir}{file_names[i]}.$(ClusterId)$(ProcId).err\n' \
                  f'log = {bash_dir}{file_names[i]}.$(ClusterId)$(ProcId).err\n' \
                  f'+JobFlavour = "tomorrow"' \
                  f'queue'

    os.system(f'echo "{sub_content}" > {sub_dir}{sub_file_names[i]}')
    os.system(f'chmod a+x {sub_dir}{sub_file_names[i]}')















