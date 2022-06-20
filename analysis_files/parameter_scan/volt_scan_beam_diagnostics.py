'''
File to analyse beam parameters as a function of voltage in the SPS simulations.

Author: Birk Emil Karlsen-Bæck
'''

# Import --------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import utility_files.analysis_tools as at
import utility_files.data_utilities as dut
from analysis_files.measurement_analysis.import_data import measured_offset

plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 16
    })


# Options -------------------------------------------------------------------------------------------------------------
PLT_SMTH = False
SCAN_MODE = 'volt'

# Parameters
extent = 0.2
n_sims = 20
number_of_batches = 4
batch_length = 72
distance = 20
until_turn = 3000
V = 6.7         # [MV]
fr = 3

tb = 4.990159369074305e-09
T_rev = 4620 / 200.394e6

# Directory and Files -------------------------------------------------------------------------------------------------
if SCAN_MODE == 'volt':
    data_folder = f'../../data_files/beam_parameters_tbt/200MHz_volt_scan_fr{fr}/'
    volt_errors = np.linspace(1 - extent, 1 + extent, n_sims)

    pos_files = dut.mk_file_names_volt_scan(volt_errors, 'pos_fit_tbt', fr=fr)
    fwhm_files = dut.mk_file_names_volt_scan(volt_errors, 'fwhm_tbt', fr=fr)
elif SCAN_MODE == 'freq':
    data_folder = f'../../data_files/beam_parameters_tbt/200MHz_volt_scan_fr{fr}/'

    TWC3_freq = 200.03766667e6
    TWC4_freq = 199.9945e6
    Design_freq = 200.222e6

    volt_errors = np.linspace(TWC4_freq, Design_freq, n_sims)

    pos_files = dut.mk_file_names_volt_scan(volt_errors, 'pos_fit_tbt', fr=fr)
    fwhm_files = dut.mk_file_names_volt_scan(volt_errors, 'fwhm_tbt', fr=fr)

# Bunch Position Analysis ---------------------------------------------------------------------------------------------
pos_data = dut.get_data_from_files(data_folder, pos_files)

normal_bunches = at.get_bunch_pos_in_buckets(1000, 5, 50, number_of_batches, batch_length)

# Find average dipole oscillations for each batch of each simulation
avg_dipole_osc = np.zeros((len(pos_files), number_of_batches, pos_data.shape[2]))
max_avg_dipole_osc = np.zeros((len(pos_files), number_of_batches))

exclude_start = 500

for i in range(len(pos_files)):
    avg_dipole_osc[i,:,:] = at.find_average_dipole_oscillation(pos_data[i,:,:], normal_bunches, until_turn, distance,
                                                               batch_length, number_of_batches)

    max_avg_dipole_osc[i,:] = np.max(avg_dipole_osc[i,:,exclude_start:], axis=1)

plt.figure()
plt.title(r'Maximum Average Dipole Oscillation, 2018 $f_r$')
V_array = V * volt_errors
y_s = 1e3
plt.plot(V_array, max_avg_dipole_osc[:,0] * y_s, label='Batch 1', color='r')
plt.plot(V_array, max_avg_dipole_osc[:,1] * y_s, label='Batch 2', color='b')
plt.plot(V_array, max_avg_dipole_osc[:,2] * y_s, label='Batch 3', color='g')
plt.plot(V_array, max_avg_dipole_osc[:,3] * y_s, label='Batch 4', color='black')
plt.legend()
if SCAN_MODE == 'volt':
    plt.xlabel('200 MHz RF Voltage [MV]')
elif SCAN_MODE == 'freq':
    plt.xlabel('200 MHz TWC $f_r$ [MHz]')
plt.ylabel('Dipole Oscillation Amplitude [ps]')
plt.xlim((V_array[0], V_array[-1]))

print(max_avg_dipole_osc[-1,0] * y_s, max_avg_dipole_osc[-1,1] * y_s,
      max_avg_dipole_osc[-1,2] * y_s, max_avg_dipole_osc[-1,3] * y_s)

print(np.mean(max_avg_dipole_osc[-1,:] * y_s))

print(np.mean(np.array([6.85, 6.93, 6.79 ])))

plt.show()



