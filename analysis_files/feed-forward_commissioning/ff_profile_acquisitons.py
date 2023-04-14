'''
Analysis of the profile acqusitions taken during the feed-forward commissioning done on 16th November 2022.

Author: Birk Emil Karlsen-Bæck
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import beam_dynamics_tools.data_visualisation.make_plots_pretty
from beam_dynamics_tools.data_management.importing_data import import_sps_profiles, \
    find_files_in_folder_starting_and_ending_with, remove_filetype_from_name, sort_sps_profiles
from beam_dynamics_tools.beam_profiles.bunch_profile_tools import bunch_by_bunch_spacing, extract_bunch_parameters

# Directories
fdir = f'../../data_files/ff_measurements/profiles/'
N_frames = 100
N_batches = 2
batch_length = 48
samples_per_file = 16800000
delta_t = 2.5e-11
shots_w_ff = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 11])
shots_wo_ff = np.array([9, 10])
choose_shots = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

# Options
PLT_PROFILES = False

# Fetching profiles
profile_files = find_files_in_folder_starting_and_ending_with(fdir, prefix='MD', suffix='.npy')
profile_files = remove_filetype_from_name(profile_files)
profile_files = sort_sps_profiles(profile_files)
profile_files = np.array(profile_files, dtype=str)
profiles, profiles_corr = import_sps_profiles(fdir,
                                              profile_files[choose_shots],
                                              N_samples_per_file=samples_per_file, prt=True)

# Analyse bunch-by-bunch position variation
print(f'Analyzing bunch-by-bunch position variations...')
spacings = np.zeros((len(profiles_corr), N_frames, N_batches, batch_length))
shot_average = np.zeros((len(profiles_corr), N_batches, batch_length))
shot_std = np.zeros((len(profiles_corr), N_batches, batch_length))
t = np.arange(int(samples_per_file/N_frames)) * delta_t

for i in tqdm(range(spacings.shape[0])):
    acq_i = profiles_corr[i, :].reshape(N_frames, int(samples_per_file/N_frames))

    if PLT_PROFILES:
        plt.figure()
        plt.plot(acq_i[0, :])
        plt.show()

    for j in range(spacings.shape[1]):
        bpos, blen, bint = extract_bunch_parameters(t, acq_i[j, :])
        spacings[i, j, :, :] = bunch_by_bunch_spacing(bpos, batch_len=batch_length)

    shot_average[i, :, :] = np.mean(spacings[i, :, :, :], axis=0)
    shot_std[i, :, :] = np.std(spacings[i, :, :, :], axis=0)


# Plots of the offsets
bnum = np.arange(batch_length)
plt.figure()
color_map = plt.get_cmap('plasma', shot_average.shape[0])
for i in range(shot_average.shape[0]):
    plt.plot(bnum, shot_average[i, 0, :] * 1e3, c=color_map(i))
    plt.fill_between(bnum, (shot_average[i, 0, :] - shot_std[i, 0, :]) * 1e3,
                     (shot_average[i, 0, :] + shot_std[i, 0, :]) * 1e3,
                     color=color_map(i), alpha=0.3)
plt.xlabel(r'Bunch number [-]')
plt.ylabel(r'$\Delta t_s$ [ps]')
plt.xlim((bnum[0], bnum[-1]))
plt.grid()


average_with_ff = np.mean(shot_average[shots_w_ff, :, :], axis=0)
std_with_ff = np.std(spacings[shots_w_ff, :, :, :], axis=(0, 1))

average_without_ff = np.mean(shot_average[shots_wo_ff, :, :], axis=0)
std_without_ff = np.std(spacings[shots_wo_ff, :, :, :], axis=(0, 1))

bnum = np.arange(batch_length)
batch_num = 0
plt.figure()
plt.plot(bnum, average_with_ff[batch_num, :] * 1e3, c='r', label='with FF')
plt.fill_between(bnum, (average_with_ff[batch_num, :] - std_with_ff[batch_num, :]) * 1e3,
                (average_with_ff[batch_num, :] + std_with_ff[batch_num, :]) * 1e3,
                 color='r', alpha=0.3)
plt.plot(bnum, average_without_ff[batch_num, :] * 1e3, c='b', label='without FF')
plt.fill_between(bnum, (average_without_ff[batch_num, :] - std_without_ff[batch_num, :]) * 1e3,
                (average_without_ff[batch_num, :] + std_without_ff[batch_num, :]) * 1e3,
                 color='b', alpha=0.3)
plt.legend()
plt.xlabel(r'Bunch number [-]')
plt.ylabel(r'$\Delta t_s$ [ps]')
plt.xlim((bnum[0], bnum[-1]))
plt.grid()




plt.show()



