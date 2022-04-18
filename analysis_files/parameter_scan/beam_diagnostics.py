'''
File to look at bunch position and length in order to search for instabilities.

Author: Birk Emil Karlsen-Bæck
'''

# Import ---------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import utility_files.analysis_tools as at
from analysis_files.measurement_analysis.import_data import measured_offset

plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 16
    })


# Options --------------------------------------------------------------------------------------------------------------
beam_parameter = 'Bunch Position'
file_name = 'pos_fit_tbt_f1_20.npy'
bunches = np.array([40])
batch_length = 72
number_of_batches = 4
until_turn = 27000
T_rev = 4620 / 200.394e6
distance = 20

choose_batch = 0

PLT_BP = True
PLT_BL = False

# Find files -----------------------------------------------------------------------------------------------------------
dir_current_file = os.path.dirname(os.path.abspath(__file__))
data_files_dir = dir_current_file[:-len('analysis_files/parameter_scan')] + 'data_files/'


until_turn //= 10


# Bunch Position -------------------------------------------------------------------------------------------------------
data = np.load(data_files_dir + file_name)
turns = np.linspace(0, 10 * data.shape[1], data.shape[1]) * T_rev

if PLT_BP:
    plt.figure()
    plt.title(beam_parameter)
    if choose_batch is not None:
        for j in range(len(bunches)):
            plt.plot(turns[:until_turn],
                        (data[choose_batch * batch_length + bunches[j] - 1, :until_turn] - data[
                            choose_batch * batch_length + bunches[j] - 1, 0]) * 1e9,
                        label=f'ba{choose_batch + 1}bu{bunches[j]}')
    else:
        for i in range(number_of_batches):
            for j in range(len(bunches)):
                plt.plot(turns[:until_turn],
                         (data[i * batch_length + bunches[j]-1, :until_turn] - data[
                             i * batch_length + bunches[j]-1,0]) * 1e9,
                         label=f'ba{i+1}bu{bunches[j]}')

    plt.xlabel('turns [s]')
    plt.ylabel('$\Delta t$ [ns]')
    plt.legend()

    plt.figure()
    plt.title(beam_parameter)
    if choose_batch is not None:
        for j in range(len(bunches)):
            data_i = (data[choose_batch * batch_length + bunches[j] - 1, :until_turn] - data[
                         choose_batch * batch_length + bunches[j] - 1, 0]) * 1e9
            line, error = at.find_amp_from_linear_regression(data_i, dist=distance)

            plt.plot(turns[:until_turn],
                     line,
                     label=f'ba{choose_batch + 1}bu{bunches[j]}')
            plt.plot(turns[:until_turn],
                     line + error,
                     label=f'ba{choose_batch + 1}bu{bunches[j]}')
            plt.plot(turns[:until_turn],
                     line - error,
                     label=f'ba{choose_batch + 1}bu{bunches[j]}')
    else:
        for i in range(number_of_batches):
            for j in range(len(bunches)):
                data_i = (data[i * batch_length + bunches[j] - 1, :until_turn] - data[
                    i * batch_length + bunches[j] - 1, 0]) * 1e9
                line, error = at.find_amp_from_linear_regression(data_i, dist=distance)

                plt.plot(turns[:until_turn],
                         line,
                         label=f'ba{i + 1}bu{bunches[j]}')
                plt.plot(turns[:until_turn],
                         line + error,
                         label=f'ba{i + 1}bu{bunches[j]}')
                plt.plot(turns[:until_turn],
                         line - error,
                         label=f'ba{i + 1}bu{bunches[j]}')

    plt.xlabel('turns [s]')
    plt.ylabel('$\Delta t$ [ns]')
    plt.legend()


# Dipole oscillation amplitude as a function of bunch number
lines = np.zeros(data[:, :until_turn].shape)
errors = np.zeros(data[:, :until_turn].shape)

for i in range(data.shape[0]):
    data_i = (data[i,:until_turn] - data[i, 0]) * 1e9
    lines[i,:], errors[i,:] = at.find_amp_from_linear_regression(data_i, dist=distance)

plt.figure()
plt.title('Dipole Oscillation Amplitude')
for i in range(number_of_batches):
    #plt.plot(np.mean(errors[i * batch_length: (i + 1) * batch_length, -30:], axis=1))
    plt.plot(np.max(errors[i * batch_length: (i + 1) * batch_length,100:], axis=1),
             label=f'batch {i + 1}')
plt.legend()


plt.figure()
plt.title('Average Dipole Oscillation')
#plt.plot(np.mean(errors, axis=0))
for i in range(number_of_batches):
    plt.plot(np.mean(errors[i * batch_length: (i + 1) * batch_length,:], axis=0),
             label=f'batch {i + 1}')
plt.legend()



plt.figure()
plt.title('Mean position')
m, ms = measured_offset()
for i in range(number_of_batches):
    # plt.plot(np.mean(errors[i * batch_length: (i + 1) * batch_length, -30:], axis=1))
    plt.plot(np.max(lines[i * batch_length: (i + 1) * batch_length, -1:], axis=1))

plt.fill_between(np.linspace(0, batch_length-1, batch_length),(m - ms), (m + ms),
                 color='b', alpha=0.3)
plt.plot(m, linestyle='--', color='b', alpha=1, label='M')


# FWHM -----------------------------------------------------------------------------------------------------------------
beam_parameter = 'Bunch Length'
file_name = 'fwhm_tbt_f1_20.npy'


data = np.load(data_files_dir + file_name)

if PLT_BL:
    plt.figure()
    plt.title(beam_parameter)
    if choose_batch is not None:
        for j in range(len(bunches)):
            data_i = (data[choose_batch * batch_length + bunches[j] - 1, :until_turn] - data[
                choose_batch * batch_length + bunches[j] - 1, 0]) * 1e9
            line, error = at.find_amp_from_linear_regression(data_i, dist=distance)

            plt.plot(turns[:until_turn],
                     line,
                     label=f'ba{choose_batch+1}bu{bunches[j]}')
            plt.plot(turns[:until_turn],
                     line + error,
                     label=f'ba{choose_batch + 1}bu{bunches[j]}')
            plt.plot(turns[:until_turn],
                     line - error,
                     label=f'ba{choose_batch + 1}bu{bunches[j]}')
    else:
        for i in range(number_of_batches):
            for j in range(len(bunches)):
                data_i = (data[i * batch_length + bunches[j] - 1, :until_turn] - data[
                    i * batch_length + bunches[j] - 1, 0]) * 1e9
                line, error = at.find_amp_from_linear_regression(data_i, dist=distance)

                plt.plot(turns[:until_turn],
                         line,
                         label=f'ba{i+1}bu{bunches[j]}')
                plt.plot(turns[:until_turn],
                         line + error,
                         label=f'ba{i + 1}bu{bunches[j]}')
                plt.plot(turns[:until_turn],
                         line - error,
                         label=f'ba{i + 1}bu{bunches[j]}')

    plt.xlabel('turns [s]')
    plt.ylabel(r'$\tau_{FWHM}$ [ns]')
    plt.legend()

    plt.figure()
    plt.title(beam_parameter)
    if choose_batch is not None:
        for j in range(len(bunches)):
            plt.plot(turns[:until_turn],
                     (data[choose_batch * batch_length + bunches[j] - 1, :until_turn] - data[
                         choose_batch * batch_length + bunches[j] - 1, 0]),
                     label=f'ba{choose_batch + 1}bu{bunches[j]}')
    else:
        for i in range(number_of_batches):
            for j in range(len(bunches)):
                plt.plot(turns[:until_turn],
                         (data[i * batch_length + bunches[j] - 1, :until_turn] - data[
                             i * batch_length + bunches[j] - 1, 0]),
                         label=f'ba{i + 1}bu{bunches[j]}')

    plt.xlabel('turns [s]')
    plt.ylabel(r'$\tau_{FWHM}$ [ns]')
    plt.legend()

plt.show()
