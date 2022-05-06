'''
File to analyse the power measurements performed november 2021.

Author: Birk Emil Karlsen-Bæck
'''


# Imports -------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import utility_files.data_utilities as dut
import utility_files.analysis_tools as at
from analysis_files.full_machine.full_machine_theoretical_estimates import theoretical_power_signal_cavity


# Options -------------------------------------------------------------------------------------------------------------
PLT_POWER = False
PLT_VANT = False
PRT_CAV_AN = False
CALC_TURN_VAR = False
PLT_CAV_VAR = True
PLT_POWER_EST = True
CAVITY = 4
n_points = 65536
HOME = False

# Parameters
h = 4620
f_c = 1 / 4.990159369074305e-09                     # [Hz]
T_rev = h / f_c

if CAVITY == 3 or CAVITY == 6:
    n_sections = 4
else:
    n_sections = 3

# Analysis ------------------------------------------------------------------------------------------------------------
# Values from Danilo
# mean
measured_voltages = np.array([0.91 * (1 + 0.03), 0.91 * (1 - 0.068), 1.53 * (1 - 0.208),
                              0.91 * (1 - 0.129), 0.91 * (1 - 0.168), 1.53 * (1 - 0.103)]) * 1e6

# error
measured_voltages_error = np.array([0.91 * 0.01, 0.91 * 0.003, 1.53 * 0.004,
                                    0.91 * 0.002, 0.91 * 0.007, 1.53 * 0.002]) * 1e6

set_voltage = np.array([0.91, 0.91, 1.53,
                        0.91, 0.91, 1.53]) * 1e6

# Directories
if HOME:
    data_folder = '../../data_files/2021-11-05/'
else:
    data_folder = '/Users/bkarlsen/cernbox/SPSOTFB_benchmark_data/data/2021-11-05/'
file_prefix = f'sps_otfb_data__all_buffers__cavity{CAVITY}__flattop__20211106_10'

# Retrieve data from CERNbox
file_names = dut.file_names_in_dir_from_prefix(data_folder, file_prefix)

power, time = at.retrieve_power(data_folder, file_names, CAVITY, n_points)
vant, time = at.retrieve_antenna_voltage(data_folder, file_names, CAVITY, n_points)

# Converting data to turn-by-turn then shot-by-shot
power_reshaped, t_reshaped = at.reshape_data(power, time[0,:], T_rev=T_rev)
vant_reshaped, t_reshaped = at.reshape_data(vant, time[0,:], T_rev=T_rev)

if PLT_POWER:
    at.plot_measurement_shots(power_reshaped, t_reshaped)
if PLT_VANT:
    at.plot_measurement_shots(vant_reshaped, t_reshaped)

# Comparison between th Voltage from measured antenna voltage, measured power from acquisitions and set point
P_set, _ = theoretical_power_signal_cavity(set_voltage[CAVITY - 1], f_c, n_sections)

# Antenna voltage from Danilos measurements
P_ant, _ = theoretical_power_signal_cavity(measured_voltages[CAVITY - 1], f_c, n_sections)
P_ant_up, _ = theoretical_power_signal_cavity(measured_voltages[CAVITY - 1] + measured_voltages_error[CAVITY - 1],
                                              f_c, n_sections)
P_ant_low, _ = theoretical_power_signal_cavity(measured_voltages[CAVITY - 1] - measured_voltages_error[CAVITY - 1],
                                              f_c, n_sections)

if np.abs(P_ant - P_ant_up) > np.abs(P_ant - P_ant_low):
    P_ant_std = np.abs(P_ant - P_ant_up)
else:
    P_ant_std = np.abs(P_ant - P_ant_low)


# Antenna voltage from acqusitions
P_antacq, _ = theoretical_power_signal_cavity(np.mean(vant_reshaped[:,:650]), f_c, n_sections)
P_antacq_up, _ = theoretical_power_signal_cavity(np.mean(vant_reshaped[:,:650]) + np.std(vant_reshaped[:,:650]),
                                                 f_c, n_sections)
P_antacq_low, _ = theoretical_power_signal_cavity(np.mean(vant_reshaped[:,:650]) - np.std(vant_reshaped[:,:650]),
                                                  f_c, n_sections)

if np.abs(P_antacq - P_antacq_up) > np.abs(P_antacq - P_antacq_low):
    P_antacq_std = np.abs(P_antacq - P_antacq_up)
else:
    P_antacq_std = np.abs(P_antacq - P_antacq_low)


if PRT_CAV_AN:
    # Print results from analysis
    print(f'------ C{CAVITY} ------')
    print(f'{n_sections}-section cavity')
    print(f'Voltages:')
    print(f'set point = {set_voltage[CAVITY -1]}')
    print(f'avg antenna = {measured_voltages[CAVITY -1]}')
    print(f'acq antenna = {np.mean(vant_reshaped[:,:650])} +- {np.std(vant_reshaped[:,:650])}')
    print(f'Power:')
    print(f'P_meas = {np.mean(power_reshaped[:,2000:])} +- {np.std(power_reshaped[:,2000:])}')
    print(f'P_set = {P_set}')
    print(f'P_ant = {P_ant} +- {P_ant_std}')
    print(f'P_antacq = {P_antacq} +- {P_antacq_std}')

#print(6.70e6 * (1 - 0.5442095845867135) / 2)
#print(6.70e6 * (0.5442095845867135) / 4)
#print(0.91 * 4 + 1.53 * 2)

if CALC_TURN_VAR:
    max_turn = np.zeros(6)
    max_shot = np.zeros(max_turn.shape)

    for i in range(6):
        print(f'Cavity C{i + 1}')
        CAVITY = i + 1

        file_prefix = f'sps_otfb_data__all_buffers__cavity{CAVITY}__flattop__20211106_10'

        # Retrieve data from CERNbox
        file_names = dut.file_names_in_dir_from_prefix(data_folder, file_prefix)
        power, time = at.retrieve_power(data_folder, file_names, CAVITY, n_points)
        power_reshaped, t_reshaped = at.reshape_data(power, time[0, :], T_rev=T_rev)

        print(power_reshaped.shape)

        var_max, var_min = at.find_turn_by_turn_variantions(power_reshaped, 3)
        print(var_max)
        print(var_min)
        max_turn[i] = np.max(np.concatenate((var_max, np.abs(var_min))))

        var_max, var_min = at.find_shot_by_shot_variantions(power_reshaped)
        print(var_max)
        print(var_min)
        max_shot[i] = np.max(np.array([var_max, np.abs(var_min)]))

    print('Max over all:')
    print('Turn-by-turn:', max_turn)
    print('Shot-by-shot:', max_shot)


if PLT_CAV_VAR:
    n_turns_per_shot = 23
    n_shots_per_cav = 3
    n_turns_per_cav = n_turns_per_shot * n_shots_per_cav
    n_points_per_turns = 2849
    n_3sec = 4
    n_4sec = 2

    all_power_turns_3sec = np.zeros((n_turns_per_shot * n_shots_per_cav * n_3sec, n_points_per_turns))
    all_power_turns_4sec = np.zeros((n_turns_per_shot * n_shots_per_cav * n_4sec, n_points_per_turns))

    power_mean_per_cav3 = np.zeros((n_3sec, n_points_per_turns))
    power_mean_per_cav4 = np.zeros((n_4sec, n_points_per_turns))

    cav3_names = np.array([1, 2, 4, 5])
    cav4_names = np.array([3, 6])

    i3 = 0
    i4 = 0

    for i in range(6):
        print(f'Cavity C{i + 1}')
        CAVITY = i + 1

        file_prefix = f'sps_otfb_data__all_buffers__cavity{CAVITY}__flattop__20211106_10'

        # Retrieve data from CERNbox
        file_names = dut.file_names_in_dir_from_prefix(data_folder, file_prefix)
        power, time = at.retrieve_power(data_folder, file_names, CAVITY, n_points)
        power_reshaped, t_reshaped = at.reshape_data(power, time[0, :], T_rev=T_rev)

        if CAVITY == 3 or CAVITY == 6:
            all_power_turns_4sec[n_turns_per_cav * i4: n_turns_per_cav * (i4 + 1), :] = power_reshaped
            power_mean_per_cav4[i4, :] = np.mean(power_reshaped, axis=0)
            i4 += 1
        else:
            all_power_turns_3sec[n_turns_per_cav * i3: n_turns_per_cav * (i3 + 1), :] = power_reshaped
            power_mean_per_cav3[i3, :] = np.mean(power_reshaped, axis=0)
            i3 += 1

    mean_3sec = np.mean(all_power_turns_3sec, axis=0)
    mean_4sec = np.mean(all_power_turns_4sec, axis=0)
    t = t_reshaped[0, :]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    P_s = 1e-3
    t_s = 1e6

    ax[0].set_title('3-section')
    ax[0].fill_between(t * t_s, P_s * mean_3sec * 0.8, P_s * mean_3sec * 1.2,
                 color='black', alpha=0.3)
    ax[0].plot(t * t_s, P_s * mean_3sec, color='black', label='Mean', linestyle='--')
    for i in range(n_3sec):
        ax[0].plot(t * t_s, P_s * power_mean_per_cav3[i, :], label=f'C{cav3_names[i]}')
    ax[0].set_xlim((3e-6 * t_s, 17e-6 * t_s))
    ax[0].set_xlabel(r'$\Delta t$ [$\mu$s]')
    ax[0].set_ylabel(r'$P$ [kW]')
    ax[0].legend()

    ax[1].set_title('4-section')
    ax[1].fill_between(t * t_s, P_s * mean_4sec * 0.8, P_s * mean_4sec * 1.2,
                       color='black', alpha=0.3)
    ax[1].plot(t * t_s, P_s * mean_4sec, color='black', label='Mean', linestyle='--')
    for i in range(n_4sec):
        ax[1].plot(t * t_s, P_s * power_mean_per_cav4[i, :], label=f'C{cav4_names[i]}')
    ax[1].set_xlim((3e-6 * t_s, 17e-6 * t_s))
    ax[1].set_xlabel(r'$\Delta t$ [$\mu$s]')
    ax[1].set_ylabel(r'$P$ [kW]')
    ax[1].legend()

if PLT_POWER_EST:
    # Plotting different power estimates

    # Mean values of estimates
    P_set = np.array([233, 233, 411, 233, 233, 411])
    P_antacq = np.array([232.0, 232.4, 405.9, 232.8, 233.1, 405.5])
    P_antmeas = np.array([247.6, 202.8, 257.6, 177.1, 161.6, 330.4])
    P_acq = np.array([244, 243, 393, 231, 235, 395])

    # Errors in estimates
    P_antacq_err = np.array([0.2, 0.2, 0.3, 0.1, 0.2, 0.3])
    P_antmeas_err = np.array([4.8, 1.3, 2.6, 0.8, 2.7, 1.5])
    P_acq_err = np.array([49, 49, 79, 46, 47, 79])

    # Cavity names
    cav_names = np.array(['C1', 'C2', 'C3', 'C4', 'C5', 'C6'])

    plt.figure()
    plt.title('Power Estimates')
    plt.errorbar(cav_names, P_acq, yerr=P_acq_err, fmt='_', color='g', label=r'$P_{acq}$', markersize=10)
    plt.plot(cav_names, P_set, '_', color='black', label=r'$P_{set}$', markersize=10)
    plt.errorbar(cav_names, P_antacq, yerr=P_antacq_err, fmt='_', color='r', label=r'$P_{ant,acq}$', markersize=10)
    #plt.plot(cav_names, P_antacq, '.', color='r', label=r'$P_{ant,acq}$')
    plt.errorbar(cav_names, P_antmeas, yerr=P_antmeas_err, fmt='_', color='b', label=r'$P_{ant,meas}$', markersize=10)
    #plt.plot(cav_names, P_antmeas, '.', color='b', label=r'$P_{ant,meas}$')
    plt.ylabel(r'Power [kW]')
    plt.xlabel(r'Cavity')


    plt.legend()



plt.show()