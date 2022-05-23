'''
File to launch simulations in lxplus with different 200 MHz voltage.

Author Birk Emil Karlsen-Bæck
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
parser.add_argument("--bunch_length", "-bl", type=float,
                    help="Option to modify bunchlength by some factor, default is 1.0")
parser.add_argument("--pl_config", "-pc", type=int,
                    help="Option to include (1) a phase loop to the simulation.")
parser.add_argument("--volt_config", "-vc", type=int,
                    help="Different values for the RF voltage.")
parser.add_argument("--gllrf_config", "-gc", type=int,
                    help="Different configurations of G_llrf for parameter scan.")
parser.add_argument("--extent", "-ex", type=float,
                    help="The range in which the voltage scan is performed, default is 0.10.")
parser.add_argument("--n_simulations", "-ns", type=int,
                    help="The amount of simulations performed in the range, default is 10.")

args = parser.parse_args()


# Parameter Values ------------------------------------------------------------
FREQ_CONFIG = 3
IMP_CONFIG = 1
VOLT_CONFIG = 1
GLLRF_CONFIG = 5
imp_str = ''
V_ERR = 1
PL_CONFIG = False
bl_factor = 1.0
n_ramp = 0
extent = 0.1
n_sims = 10
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

if args.gllrf_config is not None:
    GLLRF_CONFIG = args.gllrf_config

if args.n_ramp is not None:
    n_ramp = args.n_ramp

if args.extent is not None:
    extent = args.extent

if args.n_simulations is not None:
    n_sims = args.n_simulations

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


input_array = np.linspace(1 - extent, 1 + extent, n_sims)

gllrf_array = np.array([5, 10, 14, 16, 20])
G_llrf = gllrf_array[GLLRF_CONFIG - 1]


# Make necessary preparations for Sims ----------------------------------------

bash_file_names = np.zeros(input_array.shape).tolist()
sub_file_names = np.zeros(input_array.shape).tolist()
file_names = np.zeros(input_array.shape).tolist()

print('\nMaking shell scripts...')

for i in range(len(input_array)):
    bash_file_names[i] = f'scan_fr{FREQ_CONFIG}_vc{VOLT_CONFIG}_ve{100 * input_array[i]:.0f}{imp_str}{pl_str}{ramp_str}' \
                         f'_bl{100 * bl_factor:.0f}_llrf_{G_llrf:.0f}.sh'
    sub_file_names[i] = f'scan_fr{FREQ_CONFIG}_vc{VOLT_CONFIG}_ve{100 * input_array[i]:.0f}{imp_str}{pl_str}{ramp_str}' \
                        f'_bl{100 * bl_factor:.0f}_llrf_{G_llrf:.0f}.sub'
    file_names[i] = f'scan_fr{FREQ_CONFIG}_vc{VOLT_CONFIG}_ve{100 * input_array[i]:.0f}{imp_str}{pl_str}{ramp_str}' \
                    f'_bl{100 * bl_factor:.0f}_llrf_{G_llrf:.0f}'

    # Make bash file
    os.system(f'touch {bash_dir}{bash_file_names[i]}')

    bash_content = f'#!/bin/bash\n' \
                   f'source /afs/cern.ch/user/b/bkarlsen/.bashrc\n' \
                   f'python /afs/cern.ch/work/b/bkarlsen/Simulation_Files/SPS_OTFB/sim_files/' \
                   f'OTFB_acq_simulation_ft_real.py ' \
                   f'-nt 30000 -nr {n_ramp} -oc 1 ' \
                   f'-vc {VOLT_CONFIG} -fc {FREQ_CONFIG} ' \
                   f'-gc {GLLRF_CONFIG} -sd ' \
                   f'scan_fr{FREQ_CONFIG}_vc{VOLT_CONFIG}_ve{100 * input_array[i]:.0f}{imp_str}{pl_str}{ramp_str}' \
                   f'_bl{100 * bl_factor:.0f}_llrf_{G_llrf:.0f}/ ' \
                   f'-ve {input_array[i]} -ic {IMP_CONFIG} -bl {bl_factor} -pc {int(PL_CONFIG)}'

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
