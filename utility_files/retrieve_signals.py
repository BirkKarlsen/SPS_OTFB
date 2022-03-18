'''
File to retrieve signals for analysis in lxplus.

Author: Birk Emil Karlsen-Bæck
'''

# Import ----------------------------------------------------------------------
import numpy as np
import os
import subprocess
import argparse


# Parse arguments -------------------------------------------------------------
parser = argparse.ArgumentParser(description="This file launches simulations for parameter scans.")

parser.add_argument("--signal_type", "-st", type=str,
                    help="The signal that is being retrived from parameter scan; default is profiles")
parser.add_argument("--save_dir", "-sd", type=str,
                    help="Name of directory and file to save the signals into.")
parser.add_argument("--freq_config", "-fc", type=int,
                    help="Resonant frequency configuration for scan; default is measured (1)")
parser.add_argument("--extended", "-ext", type=int,
                    help="Option to pass one if the scan was extended or not")
parser.add_argument("--date_str", "-ds", type=str,
                    help="Option to specify the date of the simulations„ default is 'Mar-17-2022/'.")

args = parser.parse_args()

# Define parameters for script ------------------------------------------------
signal_name = 'profile_'
mst_dir = os.getcwd()[:-len('utility_files')]
FREQ_CONFIG = 1
date_string = 'Mar-17-2022/'

save_dir = mst_dir + 'data_files/'
data_dir = mst_dir + 'data_files/with_impedance/'
dir_within_sim = '30000turns_fwhm_288/sim_data/'

save_name = f'profiles_scan_fr{FREQ_CONFIG}/'
ratio_array = np.linspace(0.9, 1.1, 10)


# Arguments are introduced ----------------------------------------------------
if args.extended is not None:
    ratio_array = np.linspace(0.8, 1.2, 10)

if args.save_dir is not None:
    save_name = str(args.save_dir)

if args.signal_type is not None:
    signal_name = str(args.signal_type) + '_'

if args.date_str is not None:
    date_string = str(args.date_str)

if args.freq_config is not None:
    FREQ_CONFIG = int(args.freq_config)

# Implement the changes made by the parser
data_dir += date_string

# Retrieve data ----------------------------------------------------------------

# Make the directory that the files will be saved to
if not os.path.exists(save_dir + save_name):
    os.mkdir(save_dir + save_name)

# Search for the data within the given directory
for i in range(len(ratio_array)):
    sim_dir_i = data_dir + f'scan_fr{FREQ_CONFIG}_tr_{100 * ratio_array[i]:.0f}/' + dir_within_sim

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

    os.system(f"cp {sim_dir_i}{file_i[:-4]}_tr{100 * ratio_array[i]:.0f}.npy {save_dir + save_name}.")











