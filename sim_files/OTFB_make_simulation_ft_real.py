'''
File to make simulations using the OTFB_acq_simulation_ft_real.py simulation script.

Author: Birk Emil Karlsen-Bæck
'''


# Import --------------------------------------------------------------------------------------------------------------
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="This file launches simulations with the parameters given as parsers. "
                                             "If no parameters are given, then the base-case simulation with be "
                                             "launched.")

# Parsers related to file management
parser.add_argument("--sim_name", "--sn", type=str,
                    help="Option to give custom name to the simulation. If none is given, then a default name is given "
                         "based on the simulation configurations.")
parser.add_argument("--save_dir", "-sd", type=str,
                    help="Name of directory to save the results to.")

# Parsers that can be sent to the simulation script
parser.add_argument("--n_turns", '-nt', type=int,
                    help="The number of turns to simulates, default is 1000")
parser.add_argument("--n_ramp", "-nr", type=int,
                    help="The number of turns to track the intensity ramp, default is 5000")
parser.add_argument("--volt_config", "-vc", type=int,
                    help="Different values for the RF voltage.")
parser.add_argument("--freq_config", "-fc", type=int,
                    help="Different configurations of the TWC central frequencies.")
parser.add_argument("--gllrf_config", "-gc", type=int,
                    help="Different configurations of G_llrf for parameter scan.")
parser.add_argument("--imp_config", "-ic", type=int,
                    help="Different configurations of the impedance model for the SPS.")
parser.add_argument("--pl_config", "-pc", type=int,
                    help="Option to include (1) a phase loop to the simulation.")
parser.add_argument("--feedforward", "-ff", type=int,
                    help="Option to enable the SPS feed-forward, default is False (0).")
parser.add_argument("--fir_filter", "-fir", type=int,
                    help="Option to choose FIR filter for FF, default is only real (1).")
parser.add_argument("--tx_ratio", "-tr", type=float,
                    help="Option to tweak the optimal transmitter gain, default is 1.")
parser.add_argument("--v_error", "-ve", type=float,
                    help="Option to account for voltage error in measurements.")
parser.add_argument("--bunch_length", "-bl", type=float,
                    help="Option to modify bunchlength by some factor, default is 1.0")
parser.add_argument("--delta_freq", "-df", type=float,
                    help="Option to shift the central frequency for both cavities together.")
parser.add_argument("--more_particles", "-mp", type=int,
                    help="Option to double the amount of macro particles per bunch.")


args = parser.parse_args()

# Parameter Values ----------------------------------------------------------------------------------------------------
n_turns = 30000
n_ramp = 0
volt_config = 1
freq_config = 1
gllrf_config = 5
imp_config = 1
pl_config = 0
feedforward = 0
fir_filter = 1
tx_ratio = 1.0
v_error = 1.0
bunch_length = 1.0
delta_freq = 0
more_particles = 0

gllrf_array = np.array([5, 10, 14, 16, 20])
imp_str = ''
bash_dir = '/afs/cern.ch/work/b/bkarlsen/Simulation_Files/SPS_OTFB/bash_files/'
sub_dir = '/afs/cern.ch/work/b/bkarlsen/Submittion_Files/SPS_OTFB/'

if args.n_turns is not None:
    n_turns = args.n_turns

if args.n_ramp is not None:
    n_turns = args.n_ramp

if args.volt_config is not None:
    volt_config = args.volt_config

if args.freq_config is not None:
    freq_config = args.freq_config

if args.gllrf_config is not None:
    gllrf_config = args.gllrf_config

if args.imp_config is not None:
    imp_config = args.imp_config

if args.pl_config is not None:
    pl_config = args.pl_config

if args.feedforward is not None:
    feedforward = args.feedforward

if args.fir_filter is not None:
    fir_filter = args.fir_filter

if args.tx_ratio is not None:
    tx_ratio = args.tx_ratio

if args.v_error is not None:
    v_error = args.v_error

if args.bunch_length is not None:
    bunch_length = args.bunch_length

if args.delta_freq is not None:
    delta_freq = args.delta_freq

if args.more_particles is not None:
    more_particles = args.more_particles


if imp_config == 1:
    imp_str = ''
elif imp_config == 2:
    imp_str = '_no_otfb'
elif imp_config == 3:
    imp_str = '_only_otfb'

if bool(pl_config):
    pl_str = '_with_pl'
else:
    pl_str = ''

if n_ramp == 0:
    ramp_str = ''
else:
    ramp_str = '_ramped'

if bool(feedforward):
    ff_str = f'_with_ff{fir_filter}'
else:
    ff_str = ''

if bool(more_particles):
    mp_str = '_2xmp'
else:
    mp_str = ''

# Make necessary preparations for Sims --------------------------------------------------------------------------------


print('\nMaking shell scripts...')


bash_file_names = f'sim_fr{freq_config}_vc{volt_config}_ve{100 * v_error:.0f}' \
                  f'{imp_str}{pl_str}{ramp_str}{ff_str}{mp_str}' \
                  f'_bl{100 * bunch_length:.0f}_llrf_{gllrf_array[gllrf_config]:.0f}.sh'
sub_file_names = f'sim_fr{freq_config}_vc{volt_config}_ve{100 * v_error:.0f}' \
                 f'{imp_str}{pl_str}{ramp_str}{ff_str}{mp_str}' \
                 f'_bl{100 * bunch_length:.0f}_llrf_{gllrf_array[gllrf_config]:.0f}.sub'
file_names = f'sim_fr{freq_config}_vc{volt_config}_ve{100 * v_error:.0f}' \
             f'{imp_str}{pl_str}{ramp_str}{ff_str}{mp_str}' \
             f'_bl{100 * bunch_length:.0f}_llrf_{gllrf_array[gllrf_config]:.0f}'

save_dir = f'sim_fr{freq_config}_vc{volt_config}_ve{100 * v_error:.0f}' \
           f'{imp_str}{pl_str}{ramp_str}{ff_str}{mp_str}' \
           f'_bl{100 * bunch_length:.0f}_llrf_{gllrf_array[gllrf_config]:.0f}/'

if args.sim_name is not None:
    bash_file_names = args.sim_name + '.sh'
    sub_file_names = args.sim_name + '.sub'
    file_names = args.sim_name

if args.save_dir is not None:
    save_dir = args.save_dir


# Make bash file
os.system(f'touch {bash_dir}{bash_file_names}')

bash_content = f'#!/bin/bash\n' \
               f'source /afs/cern.ch/user/b/bkarlsen/.bashrc\n' \
               f'python /afs/cern.ch/work/b/bkarlsen/Simulation_Files/SPS_OTFB/sim_files/' \
               f'OTFB_acq_simulation_ft_real.py ' \
               f'-nt {n_turns} -nr {n_ramp} -vc {volt_config} -fc {freq_config} -gc {gllrf_config} ' \
               f'-ic {imp_config} -pc {pl_config} -ff {feedforward} -fir {fir_filter} ' \
               f'-tr {tx_ratio} -ve {v_error} -bl {bunch_length} -df {delta_freq} -mp {more_particles} ' \
               f'-sd {save_dir} ' \

os.system(f'echo "{bash_content}" > {bash_dir}{bash_file_names}')
os.system(f'chmod a+x {bash_dir}{bash_file_names}')


print('\nMaking and submitting simulations...')

# Make submission file
os.system(f'touch {sub_dir}{sub_file_names}')

sub_content = f'executable = {bash_dir}{bash_file_names}\n' \
              f'arguments = \$(ClusterId)\$(ProcId)\n' \
              f'output = {bash_dir}{file_names}.\$(ClusterId)\$(ProcId).out\n' \
              f'error = {bash_dir}{file_names}.\$(ClusterId)\$(ProcId).err\n' \
              f'log = {bash_dir}{file_names}.\$(ClusterId)\$(ProcId).log\n' \
              f'+JobFlavour = \\"testmatch\\"\n' \
              f'queue'

os.system(f'echo "{sub_content}" > {sub_dir}{sub_file_names}')
os.system(f'chmod a+x {sub_dir}{sub_file_names}')

os.system(f'condor_submit {sub_dir}{sub_file_names}')
