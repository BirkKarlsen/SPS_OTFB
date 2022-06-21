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
PLT_ALL_FRS = True
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
ve = 89
omit_sim = None#np.array([18])

tb = 4.990159369074305e-09
T_rev = 4620 / 200.394e6

# Directory and Files -------------------------------------------------------------------------------------------------
if SCAN_MODE == 'volt':
    if PLT_ALL_FRS:
        frs = np.array([1, 2, 3])
        pos_files = []
        fwhm_files = []
        for i in range(len(frs)):
            data_folder = f'../../data_files/beam_parameters_tbt/200MHz_volt_scan_fr{frs[i]}/'
            volt_errors = np.linspace(1 - extent, 1 + extent, n_sims)

            pos_files.append(dut.mk_file_names_volt_scan(volt_errors, 'pos_fit_tbt', fr=frs[i]))
            fwhm_files.append(dut.mk_file_names_volt_scan(volt_errors, 'fwhm_tbt', fr=frs[i]))

        pos_files = np.array(pos_files)
        fwhm_files = np.array(fwhm_files)
    else:
        data_folder = f'../../data_files/beam_parameters_tbt/200MHz_volt_scan_fr{fr}/'
        volt_errors = np.linspace(1 - extent, 1 + extent, n_sims)

        pos_files = dut.mk_file_names_volt_scan(volt_errors, 'pos_fit_tbt', fr=fr)
        fwhm_files = dut.mk_file_names_volt_scan(volt_errors, 'fwhm_tbt', fr=fr)

elif SCAN_MODE == 'freq':
    data_folder = f'../../data_files/beam_parameters_tbt/200MHz_freq_scan_ve{ve}/'

    TWC3_freq = 200.03766667e6
    TWC4_freq = 199.9945e6
    Design_freq = 200.222e6

    volt_errors = np.linspace(TWC4_freq, Design_freq, n_sims)

    pos_files = dut.mk_file_names_freq_scan(volt_errors, 'pos_fit_tbt', fr=fr)
    fwhm_files = dut.mk_file_names_freq_scan(volt_errors, 'fwhm_tbt', fr=fr)
else:
    print('In valid scan mode!')

# Bunch Position Analysis ---------------------------------------------------------------------------------------------
if PLT_ALL_FRS and SCAN_MODE == 'volt':
    pos_data = []

    for i in range(len(frs)):
        data_folder = f'../../data_files/beam_parameters_tbt/200MHz_volt_scan_fr{frs[i]}/'
        pos_data.append(dut.get_data_from_files(data_folder, pos_files[i, :]))

    pos_data = np.array(pos_data)

else:
    pos_data = dut.get_data_from_files(data_folder, pos_files)

normal_bunches = at.get_bunch_pos_in_buckets(1000, 5, 50, number_of_batches, batch_length)

# Find average dipole oscillations for each batch of each simulation

fig, ax = plt.subplots()
ax.set_title(r'Maximum Average Dipole Oscillation')
if PLT_ALL_FRS and SCAN_MODE == 'volt':
    ln_stl = [None, '--', ':']
    for j in range(len(frs)):
        pos_dataj = pos_data[j, :, :, :]
        avg_dipole_osc = np.zeros((pos_files.shape[1], number_of_batches, pos_dataj.shape[2]))
        max_avg_dipole_osc = np.zeros((pos_files.shape[1], number_of_batches))

        exclude_start = 500

        for i in range(pos_files.shape[1]):
            avg_dipole_osc[i, :, :] = at.find_average_dipole_oscillation(pos_dataj[i, :, :], normal_bunches, until_turn,
                                                                         distance,
                                                                         batch_length, number_of_batches)

            max_avg_dipole_osc[i, :] = np.max(avg_dipole_osc[i, :, exclude_start:], axis=1)


        V_array = V * volt_errors


        if omit_sim is not None:
            max_avg_dipole_osc1 = np.delete(max_avg_dipole_osc[:, 0], omit_sim)
            max_avg_dipole_osc2 = np.delete(max_avg_dipole_osc[:, 1], omit_sim)
            max_avg_dipole_osc3 = np.delete(max_avg_dipole_osc[:, 2], omit_sim)
            max_avg_dipole_osc4 = np.delete(max_avg_dipole_osc[:, 3], omit_sim)
            V_array = np.delete(V_array, omit_sim)
        else:
            max_avg_dipole_osc1 = max_avg_dipole_osc[:, 0]
            max_avg_dipole_osc2 = max_avg_dipole_osc[:, 1]
            max_avg_dipole_osc3 = max_avg_dipole_osc[:, 2]
            max_avg_dipole_osc4 = max_avg_dipole_osc[:, 3]

        y_s = 1e3
        if j == 0:
            labels = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4']
        else:
            labels = [None, None, None, None]

        ax.plot(V_array, max_avg_dipole_osc1 * y_s, label=labels[0], color='r', linestyle=ln_stl[j])
        ax.plot(V_array, max_avg_dipole_osc2 * y_s, label=labels[1], color='b', linestyle=ln_stl[j])
        ax.plot(V_array, max_avg_dipole_osc3 * y_s, label=labels[2], color='g', linestyle=ln_stl[j])
        ax.plot(V_array, max_avg_dipole_osc4 * y_s, label=labels[3], color='black', linestyle=ln_stl[j])
        if j == 0:
            lines = ax.get_lines()
            legend1 = plt.legend(lines, labels, loc=1)


        ax.set_xlabel('200 MHz RF Voltage [MV]')
        ax.set_ylabel('Dipole Oscillation Amplitude [ps]')
        ax.set_xlim((V_array[0], V_array[-1]))

    dummy_lines = []
    labels_sim = [r'$f_r$ 2021', r'$f_r$ Design', r'$f_r$ 2018']
    for i in range(len(ln_stl)):
        dummy_lines.append(ax.plot([], [], label=labels_sim[i], color='black', linestyle=ln_stl[i]))

    plt.legend(loc=1, ncol=2)
    #legend2 = plt.legend(dummy_lines, [r'$f_r$ 2021', r'$f_r$ Design', r'$f_r$ 2018'], loc=4)
    #ax.add_artist(legend1)
else:
    avg_dipole_osc = np.zeros((len(pos_files), number_of_batches, pos_data.shape[2]))
    max_avg_dipole_osc = np.zeros((len(pos_files), number_of_batches))

    exclude_start = 500

    for i in range(len(pos_files)):
        avg_dipole_osc[i,:,:] = at.find_average_dipole_oscillation(pos_data[i,:,:], normal_bunches, until_turn, distance,
                                                                   batch_length, number_of_batches)

        max_avg_dipole_osc[i,:] = np.max(avg_dipole_osc[i,:,exclude_start:], axis=1)

    if SCAN_MODE == 'volt':
        V_array = V * volt_errors
    elif SCAN_MODE == 'freq':
        V_array = volt_errors * 1e-6

    if omit_sim is not None:
        max_avg_dipole_osc1 = np.delete(max_avg_dipole_osc[:,0], omit_sim)
        max_avg_dipole_osc2 = np.delete(max_avg_dipole_osc[:,1], omit_sim)
        max_avg_dipole_osc3 = np.delete(max_avg_dipole_osc[:,2], omit_sim)
        max_avg_dipole_osc4 = np.delete(max_avg_dipole_osc[:,3], omit_sim)
        V_array = np.delete(V_array, omit_sim)
    else:
        max_avg_dipole_osc1 = max_avg_dipole_osc[:, 0]
        max_avg_dipole_osc2 = max_avg_dipole_osc[:, 1]
        max_avg_dipole_osc3 = max_avg_dipole_osc[:, 2]
        max_avg_dipole_osc4 = max_avg_dipole_osc[:, 3]

    y_s = 1e3
    plt.plot(V_array, max_avg_dipole_osc1 * y_s, label='Batch 1', color='r')
    plt.plot(V_array, max_avg_dipole_osc2 * y_s, label='Batch 2', color='b')
    plt.plot(V_array, max_avg_dipole_osc3 * y_s, label='Batch 3', color='g')
    plt.plot(V_array, max_avg_dipole_osc4 * y_s, label='Batch 4', color='black')
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



