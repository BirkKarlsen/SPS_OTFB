'''
File to retrieve files from voltage scans and put them into a folder.

Author: Birk Emil Karlsen-Bæck
'''

# Import --------------------------------------------------------------------------------------------------------------
import numpy as np
import os
import argparse


# Parse arguments -----------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="File to retrieve files from voltage scans and put them into a folder.")

# Arguments related to retrieving and saving
parser.add_argument("--signal_type", "-st", type=str,
                    help="The signal that is being retrived from parameter scan; default is profiles")
parser.add_argument("--save_dir", "-sd", type=str,
                    help="Name of directory and file to save the signals into.")
parser.add_argument("--date_str", "-ds", type=str,
                    help="Option to specify the date of the simulations„ default is 'Mar-17-2022/'.")
parser.add_argument("--tbt_param", "-tb", type=int,
                    help="Option to determine whether or not it is a turn-by-turn signal, default is true (1)")

# Arguments related to finding the specific set of simulations
parser.add_argument("--freq_config", "-fc", type=int,
                    help="Resonant frequency configuration for scan; default is measured (1)")
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

# Define parameters for script ----------------------------------------------------------------------------------------
signal_name = 'profile_'
mst_dir = os.getcwd()[:-len('utility_files')]
TBT_PARAM = True
FREQ_CONFIG = 1
date_string = 'Mar-17-2022/'
IMP_CONFIG = 1
VOLT_CONFIG = 1
GLLRF_CONFIG = 5
imp_str = ''
PL_CONFIG = False
bl_factor = 1.0
n_ramp = 0
extent = 0.1
n_sims = 10

save_dir = mst_dir + 'data_files/'
data_dir = mst_dir + 'data_files/with_impedance/'
dir_within_sim = '30000turns_fwhm_288/sim_data/'

save_name = f'profiles_scan_fr{FREQ_CONFIG}/'
input_array = np.array([1, 2, 3, 4, 5])
gain_array = np.array([5, 10, 14, 16, 20])


# Arguments are introduced --------------------------------------------------------------------------------------------
if args.save_dir is not None:
    save_name = str(args.save_dir)

if args.signal_type is not None:
    signal_name = str(args.signal_type)

if args.date_str is not None:
    date_string = str(args.date_str)

if args.freq_config is not None:
    FREQ_CONFIG = int(args.freq_config)

if args.imp_config is not None:
    IMP_CONFIG = args.imp_config

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

if args.tbt_param is not None:
    TBT_PARAM = bool(args.tbt_param)

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


# Implement the changes made by the parser
data_dir += date_string

# Retrieve data -------------------------------------------------------------------------------------------------------

# Make the directory that the files will be saved to
if not os.path.exists(save_dir + save_name):
    os.mkdir(save_dir + save_name)

# Search for the data within the given directory
for i in range(len(input_array)):
    sim_folder_i = f'scan_fr{FREQ_CONFIG}_vc{VOLT_CONFIG}_ve{100 * input_array[i]:.0f}{imp_str}{pl_str}{ramp_str}' \
                   f'_bl{100 * bl_factor:.0f}_llrf_{G_llrf:.0f}/'
    sim_dir_i = data_dir + sim_folder_i + dir_within_sim

    if TBT_PARAM:
        # Find all files for the given signal type prefix
        file_list = []
        turns = []
        for file in os.listdir(sim_dir_i[:-1]):
            if file.startswith(signal_name):
                file_list.append(file)
                turns.append(file[len(signal_name):-4])
        turns = np.array(turns, dtype=int)

        # Find latest turn that the signals was recorded and saved
        final_index = np.where(turns == np.amax(turns))[0][0]
        file_i = file_list[final_index]
    else:
        for file in os.listdir(sim_dir_i[:-1]):
            if file.startswith(signal_name):
                file_i = file

    os.system(f"cp {sim_dir_i}{file_i} {save_dir + save_name}{file_i[:-4]}"
              f"_fr{FREQ_CONFIG}_vc{VOLT_CONFIG}_ve{100 * input_array[i]:.0f}"
              f"{imp_str}{pl_str}{ramp_str}_bl{100 * bl_factor:.0f}_llrf_{G_llrf:.0f}.npy")
