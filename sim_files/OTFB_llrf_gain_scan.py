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
parser.add_argument("--n_ramp", "-nr", type=int,
                    help="The number of turns to track the intensity ramp, default is 0")
parser.add_argument("--imp_config", "-ic", type=int,
                    help="Different configurations of the impedance model for the SPS.")
parser.add_argument("--v_error", "-ve", type=float,
                    help="Option to account for voltage error in measurements.")
parser.add_argument("--bunch_length", "-bl", type=float,
                    help="Option to modify bunchlength by some factor, default is 1.0")
parser.add_argument("--pl_config", "-pc", type=int,
                    help="Option to include (1) a phase loop to the simulation.")
parser.add_argument("--volt_config", "-vc", type=int,
                    help="Different values for the RF voltage.")


args = parser.parse_args()

# Parameter Values ------------------------------------------------------------
input_array = np.array([1, 2, 3, 4, 5])
gllrf_array = np.array([5, 10, 14, 16, 20])
FREQ_CONFIG = 3
IMP_CONFIG = 1
VOLT_CONFIG = 1
imp_str = ''
V_ERR = 1
PL_CONFIG = False
bl_factor = 1.0
n_ramp = 0
bash_dir = '/afs/cern.ch/work/b/bkarlsen/Simulation_Files/SPS_OTFB/bash_files/'
sub_dir = '/afs/cern.ch/work/b/bkarlsen/Submittion_Files/SPS_OTFB/'

if args.freq_config is not None:
    FREQ_CONFIG = args.freq_config

if args.imp_config is not None:
    IMP_CONFIG = args.imp_config

if args.v_error is not None:
    V_ERR = args.v_error

if args.bunch_length is not None:
    bl_factor = args.bunch_length

if args.pl_config is not None:
    PL_CONFIG = bool(args.pl_config)

if args.volt_config is not None:
    VOLT_CONFIG = args.volt_config

if args.n_ramp is not None:
    n_ramp = args.n_ramp

if IMP_CONFIG == 1:
    imp_str = ''
elif IMP_CONFIG == 2:
    imp_str = '_no_otfb'
elif IMP_CONFIG == 3:
    imp_str = '_only_otfb'

if PL_CONFIG:
    pl_str = '_with_pl'
else:
    pl_str = ''

if n_ramp == 0:
    ramp_str = ''
else:
    ramp_str = '_ramped'

# Make necessary preparations for Sims ----------------------------------------

bash_file_names = np.zeros(input_array.shape).tolist()
sub_file_names = np.zeros(input_array.shape).tolist()
file_names = np.zeros(input_array.shape).tolist()

print('\nMaking shell scripts...')

for i in range(len(input_array)):
    bash_file_names[i] = f'scan_fr{FREQ_CONFIG}_vc{VOLT_CONFIG}_ve{100 * V_ERR:.0f}{imp_str}{pl_str}{ramp_str}' \
                         f'_bl{100 * bl_factor:.0f}_llrf_{gllrf_array[i]:.0f}.sh'
    sub_file_names[i] = f'scan_fr{FREQ_CONFIG}_vc{VOLT_CONFIG}_ve{100 * V_ERR:.0f}{imp_str}{pl_str}{ramp_str}' \
                        f'_bl{100 * bl_factor:.0f}_llrf_{gllrf_array[i]:.0f}.sub'
    file_names[i] = f'scan_fr{FREQ_CONFIG}_vc{VOLT_CONFIG}_ve{100 * V_ERR:.0f}{imp_str}{pl_str}{ramp_str}' \
                    f'_bl{100 * bl_factor:.0f}_llrf_{gllrf_array[i]:.0f}'

    # Make bash file
    os.system(f'touch {bash_dir}{bash_file_names[i]}')

    bash_content = f'#!/bin/bash\n' \
                   f'source /afs/cern.ch/user/b/bkarlsen/.bashrc\n' \
                   f'python /afs/cern.ch/work/b/bkarlsen/Simulation_Files/SPS_OTFB/sim_files/' \
                   f'OTFB_acq_simulation_ft_real.py ' \
                   f'-nt 30000 -nr {n_ramp} ' \
                   f'-vc {VOLT_CONFIG} -fc {FREQ_CONFIG} ' \
                   f'-gc {input_array[i]} -sd ' \
                   f'scan_fr{FREQ_CONFIG}_vc{VOLT_CONFIG}_ve{100 * V_ERR:.0f}{imp_str}{pl_str}{ramp_str}' \
                   f'_bl{100 * bl_factor:.0f}_llrf_{gllrf_array[i]:.0f}/ ' \
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
