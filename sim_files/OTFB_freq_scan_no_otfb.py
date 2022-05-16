'''
File to simulations without the OTFB and with the three different frequency configurations.

Author: Birk Emil Karlsen-Bæck
'''

# Imports -------------------------------------------------------------------------------------------------------------
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="This file launches simulations for parameter scans.")

parser.add_argument("--freq_config", "-fc", type=int,
                    help="Different configurations of the TWC central frequencies.")

args = parser.parse_args()

# Parameter Values ----------------------------------------------------------------------------------------------------
input_array = np.array([1, 2, 3])
V_ERR = 0.89
VOLT_CONFIG = 1
PL_CONFIG = 0
IMP_CONFIG = 2

imp_str = ''
pl_str = ''
ramp_str = ''
bl_factor = 1.0
gllrf = 20
gc = 5
n_ramp = 0

bash_dir = '/afs/cern.ch/work/b/bkarlsen/Simulation_Files/SPS_OTFB/bash_files/'
sub_dir = '/afs/cern.ch/work/b/bkarlsen/Submittion_Files/SPS_OTFB/'


if IMP_CONFIG == 1:
    imp_str = ''
elif IMP_CONFIG == 2:
    imp_str = '_no_otfb'
elif IMP_CONFIG == 3:
    imp_str = '_only_otfb'


# Make necessary preparations for Sims ----------------------------------------

bash_file_names = np.zeros(input_array.shape).tolist()
sub_file_names = np.zeros(input_array.shape).tolist()
file_names = np.zeros(input_array.shape).tolist()

print('\nMaking shell scripts...')

for i in range(len(input_array)):
    bash_file_names[i] = f'scan_fr{input_array[i]}_vc{VOLT_CONFIG}_ve{100 * V_ERR:.0f}{imp_str}{pl_str}{ramp_str}' \
                         f'_bl{100 * bl_factor:.0f}_llrf_{gllrf:.0f}.sh'
    sub_file_names[i] = f'scan_fr{input_array[i]}_vc{VOLT_CONFIG}_ve{100 * V_ERR:.0f}{imp_str}{pl_str}{ramp_str}' \
                        f'_bl{100 * bl_factor:.0f}_llrf_{gllrf:.0f}.sub'
    file_names[i] = f'scan_fr{input_array[i]}_vc{VOLT_CONFIG}_ve{100 * V_ERR:.0f}{imp_str}{pl_str}{ramp_str}' \
                    f'_bl{100 * bl_factor:.0f}_llrf_{gllrf:.0f}'

    # Make bash file
    os.system(f'touch {bash_dir}{bash_file_names[i]}')

    bash_content = f'#!/bin/bash\n' \
                   f'source /afs/cern.ch/user/b/bkarlsen/.bashrc\n' \
                   f'python /afs/cern.ch/work/b/bkarlsen/Simulation_Files/SPS_OTFB/sim_files/' \
                   f'OTFB_acq_simulation_ft_real.py ' \
                   f'-nt 30000 -nr {n_ramp} -oc 1 ' \
                   f'-vc {VOLT_CONFIG} -fc {input_array[i]} ' \
                   f'-gc {gc} -sd ' \
                   f'scan_fr{input_array[i]}_vc{VOLT_CONFIG}_ve{100 * V_ERR:.0f}{imp_str}{pl_str}{ramp_str}' \
                   f'_bl{100 * bl_factor:.0f}_llrf_{gllrf:.0f}/ ' \
                   f'-ve {V_ERR} -ic {IMP_CONFIG} -bl {bl_factor} -pc {int(PL_CONFIG)}'

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
                  f'+JobFlavour = \\"testmatch\\"\n' \
                  f'queue'

    os.system(f'echo "{sub_content}" > {sub_dir}{sub_file_names[i]}')
    os.system(f'chmod a+x {sub_dir}{sub_file_names[i]}')

    os.system(f'condor_submit {sub_dir}{sub_file_names[i]}')

