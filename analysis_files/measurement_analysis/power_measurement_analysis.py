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
PLT_POWER = True
CAVITY = 6
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
at.plot_measurement_shots(power_reshaped, t_reshaped)
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

plt.figure()
plt.plot(vant_reshaped[0,:])

plt.show()