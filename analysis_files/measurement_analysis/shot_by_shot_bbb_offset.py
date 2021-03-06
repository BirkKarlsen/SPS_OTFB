'''
File to analyse the bunch-by-bunch offset shot-by-shot.

Author: Birk Emil Karlsen-Bæck
'''

# Import --------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import utility_files.data_utilities as dut
import utility_files.analysis_tools as at


# Options -------------------------------------------------------------------------------------------------------------
HOME = True
TURN_BY_TURN = True
PLT_TBT = False
SHOT_BY_SHOT = True
PLT_SBS = True

# Parameters ----------------------------------------------------------------------------------------------------------
n_points = 60
turn_points = 99999
n_turns = 100
sample_rate = 10e9
sample_period = 1 / sample_rate
N_batches = 4

# Functions -----------------------------------------------------------------------------------------------------------
if HOME:
    meas_dir = f'../../data_files/profiles_SPS_OTFB_flattop/'
else:
    meas_dir = f'/Users/bkarlsen/cernbox/SPSOTFB_benchmark_data/data/2021-11-05/profiles_SPS_OTFB_flattop/'

files = []
for i in range(104, 130):
    if i != 112:
        files.append(f'MD_{i}')

profiles_data, profiles_data_corr = dut.import_profiles(meas_dir, files)
time_array = np.linspace(0, (turn_points - 1) * sample_period, turn_points)

sbs_bbb_offset_b1 = np.zeros((len(files), 72))
sbs_bbb_offset_b2 = np.zeros((len(files), 72))
sbs_bbb_offset_b3 = np.zeros((len(files), 72))
sbs_bbb_offset_b4 = np.zeros((len(files), 72))

if TURN_BY_TURN:
    bbb_offsets_1t = np.zeros((n_turns * profiles_data.shape[0], 72))
    bbb_offsets_2t = np.zeros((n_turns * profiles_data.shape[0], 72))
    bbb_offsets_3t = np.zeros((n_turns * profiles_data.shape[0], 72))
    bbb_offsets_4t = np.zeros((n_turns * profiles_data.shape[0], 72))

    for i in range(profiles_data.shape[0]):
        profile_data_corr = profiles_data_corr[i, :]
        profile_data_corr = profile_data_corr.reshape((n_turns, turn_points))

        N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit, Bunch_peaksFit, Bunch_exponent, Goodness_of_fit = \
            dut.getBeamPattern_4(time_array, profile_data_corr.T, distance=200, fit_option='fwhm', plot_fit=False,
                                 baseline_length=35)

        bbb_offsets_1 = np.zeros((n_turns, 72))
        bbb_offsets_2 = np.zeros((n_turns, 72))
        bbb_offsets_3 = np.zeros((n_turns, 72))
        bbb_offsets_4 = np.zeros((n_turns, 72))

        for j in range(n_turns):
            pos_fit = Bunch_positionsFit[j, :].reshape((N_batches, 72))

            bbb_offsets_1[j, :] = at.find_offset(pos_fit[0, :])
            bbb_offsets_2[j, :] = at.find_offset(pos_fit[1, :])
            bbb_offsets_3[j, :] = at.find_offset(pos_fit[2, :])
            bbb_offsets_4[j, :] = at.find_offset(pos_fit[3, :])

        bbb_offsets_1t[i * n_turns: (i + 1) * n_turns, :] = bbb_offsets_1
        bbb_offsets_2t[i * n_turns: (i + 1) * n_turns, :] = bbb_offsets_2
        bbb_offsets_3t[i * n_turns: (i + 1) * n_turns, :] = bbb_offsets_3
        bbb_offsets_4t[i * n_turns: (i + 1) * n_turns, :] = bbb_offsets_4

        PLT_B_43 = True
        if PLT_B_43 and i == 8:
            plt.figure()
            plt.title('Bunch 43, shot 9')
            plt.plot(Bunch_positionsFit[:, 43])

        if PLT_TBT:
            plt.figure()
            plt.title('batch 1')
            plt.plot(bbb_offsets_1.T)

            plt.figure()
            plt.title('batch 2')
            plt.plot(bbb_offsets_2.T)

            plt.figure()
            plt.title('batch 3')
            plt.plot(bbb_offsets_3.T)

            plt.figure()
            plt.title('batch 4')
            plt.plot(bbb_offsets_4.T)

            plt.show()

        if SHOT_BY_SHOT:
            sbs_bbb_offset_b1[i, :] = np.mean(bbb_offsets_1, axis=0)
            sbs_bbb_offset_b2[i, :] = np.mean(bbb_offsets_2, axis=0)
            sbs_bbb_offset_b3[i, :] = np.mean(bbb_offsets_3, axis=0)
            sbs_bbb_offset_b4[i, :] = np.mean(bbb_offsets_4, axis=0)

    print(bbb_offsets_1t.shape)
    print(profiles_data.shape[0])

    print('Turn-by-turn')
    print('Batch 1')
    var_max, var_min, max_ind, min_ind = at.find_turn_by_turn_variantions(bbb_offsets_1t, profiles_data.shape[0],
                                                                          ABS=True, IND_MAX=True)
    print(var_max, var_min, max_ind, min_ind)
    print(np.max(np.concatenate((var_max, np.abs(var_min)))))
    at.plot_bunch_pos(bbb_offsets_1t, 100, 8, 43)

    print('Batch 2')
    var_max, var_min = at.find_turn_by_turn_variantions(bbb_offsets_2t, profiles_data.shape[0], ABS=True)
    print(var_max, var_min)
    print(np.max(np.concatenate((var_max, np.abs(var_min)))))

    print('Batch 3')
    var_max, var_min = at.find_turn_by_turn_variantions(bbb_offsets_3t, profiles_data.shape[0], ABS=True)
    print(var_max, var_min)
    print(np.max(np.concatenate((var_max, np.abs(var_min)))))

    print('Batch 4')
    var_max, var_min = at.find_turn_by_turn_variantions(bbb_offsets_4t, profiles_data.shape[0], ABS=True)
    print(var_max, var_min)
    print(np.max(np.concatenate((var_max, np.abs(var_min)))))

    print('Shot-by-shot')
    print('Batch 1')
    var_max, var_min = at.find_shot_by_shot_variantions(bbb_offsets_1t, ABS=True)
    print(var_max, var_min)
    print(np.max(np.array([var_max, np.abs(var_min)])))

    print('Batch 2')
    var_max, var_min = at.find_shot_by_shot_variantions(bbb_offsets_2t, ABS=True)
    print(var_max, var_min)
    print(np.max(np.array([var_max, np.abs(var_min)])))

    print('Batch 3')
    var_max, var_min = at.find_shot_by_shot_variantions(bbb_offsets_3t, ABS=True)
    print(var_max, var_min)
    print(np.max(np.array([var_max, np.abs(var_min)])))

    print('Batch 4')
    var_max, var_min = at.find_shot_by_shot_variantions(bbb_offsets_4t, ABS=True)
    print(var_max, var_min)
    print(np.max(np.array([var_max, np.abs(var_min)])))

if PLT_SBS and SHOT_BY_SHOT:
    plt.figure()
    plt.title('batch 1')
    plt.plot(sbs_bbb_offset_b1.T)

    plt.figure()
    plt.title('batch 2')
    plt.plot(sbs_bbb_offset_b2.T)

    plt.figure()
    plt.title('batch 3')
    plt.plot(sbs_bbb_offset_b3.T)

    plt.figure()
    plt.title('batch 4')
    plt.plot(sbs_bbb_offset_b4.T)

    plt.show()