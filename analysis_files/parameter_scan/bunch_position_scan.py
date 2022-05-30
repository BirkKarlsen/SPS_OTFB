'''
File to compare bunch-by-bunch offset across different transmitter gain values.

Author: Birk Emil Karlsen-Bæck
'''


# Imports ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
import utility_files.data_utilities as dut
import utility_files.analysis_tools as at
from analysis_files.measurement_analysis.sps_cable_transferfunction_script import cables_tranfer_function

plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 16
    })

# Measurement function --------------------------------------------------------
def measured_offset():
    pos_tot = np.load('../../data_files/beam_measurements/bunch_positions_total_red.npy')
    pos_fl = pos_tot.reshape((25 * 100, 288))
    pos_fb = pos_fl[:,:72]
    b_n = np.linspace(1, 72, 72)
    pds = np.zeros(pos_fb.shape)

    for i in range(pos_fb.shape[0]):
        s1, i1, rval, pval, stderr = linregress(b_n, pos_fb[i,:])

        pds[i,:] = pos_fb[i,:] - s1 * b_n - i1

    avg_pd = np.mean(pds, axis = 0)
    std_pd = np.std(pds, axis = 0)

    return avg_pd, std_pd


# Options ---------------------------------------------------------------------
FREQ_CONFIG = 1
EXTENDED = False
MODE = 1                    # MODE 1 is transmitter gain, MODE 2 is LLRF
omit_ind = 10

# Plots
PLT_BBB = False
PLT_BUNCH_VAR_OVER_TURNS = True

# Directories -----------------------------------------------------------------
mst_dir = os.getcwd()[:-len('analysis_files/parameter_scan')]

if MODE == 1:
    data_folder = f'tx_scan_fr{FREQ_CONFIG}_vc1_ve100_bl100_g20/'
else:
    data_folder = f'profile_scan_llrf_fr{FREQ_CONFIG}/'
data_dir = mst_dir + 'data_files/beam_parameters_tbt/' + data_folder

if EXTENDED:
    ratio_array = np.linspace(0.8, 1.2, 10) * 100
else:
    ratio_array = np.linspace(0.9, 1.1, 10) * 100

if MODE == 1:
    Ns = 10
else:
    ratio_array = np.array([5, 10, 14, 16, 20])
    Ns = 5


# Get data --------------------------------------------------------------------
if MODE == 1:
    sample_data = np.load(data_dir + f'pos_fit_tbt_tr{ratio_array[0]:.0f}.npy')
else:
    sample_data = np.load(data_dir + f'profile_29000_llrf{ratio_array[0]:.0f}.npy')

pos = np.zeros((sample_data.shape[0], sample_data.shape[1], len(ratio_array)))

print('Fetching profiles...\n')
for i in range(len(ratio_array)):
    for file in os.listdir(data_dir[:-1]):
        if MODE == 1:
            if file.endswith(f'tr{ratio_array[i]:.0f}.npy') and file.startswith('pos_fit'):
                sample_i = np.load(data_dir + file)
                pos[:, :, i] = sample_i
        else:
            if file.endswith(f'llrf{ratio_array[i]:.0f}.npy'):
                sample_i = np.load(data_dir + file)
                pos[:, :, i] = sample_i


bbb_offsets = np.zeros(pos[:72, -1, :].shape)
xs = np.zeros(pos[:72, -1, :].shape)

print('Computing bunch-by-bunch offset...\n')
for i in range(len(ratio_array)):
    bbb_offsets[:, i] = at.find_offset(pos[:72, -1, i])
    xs[:, i] = np.linspace(0, len(bbb_offsets[:, i]), len(bbb_offsets[:, i]))


bbb_offsets = bbb_offsets[:,:omit_ind] * 1e9
xs = xs[:,:omit_ind]

if PLT_BBB:
    m, ms = measured_offset()
    colormap = plt.cm.gist_ncar
    plt.figure()
    plt.title(f'Bunch-by-bunch Offset for $f_c =$ 200.1 MHz')
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, Ns))))
    plt.fill_between(xs[:,0], (m - ms) * 1e3, (m + ms) * 1e3,
                     color='b', alpha=0.3)
    plt.plot(xs[:,0], m * 1e3, linestyle='--', color='b', alpha=1, label='M')
    if MODE == 2:
        for i in range(len(ratio_array)):
            plt.plot(xs[:,i], bbb_offsets[:,i] * 1e3, label=f'{ratio_array[i]}')
    else:
        for i in range(len(ratio_array)):
            plt.plot(xs[:,i], bbb_offsets[:,i] * 1e3)
    plt.ylabel(f'$\Delta t$ [ps]')
    plt.xlabel(f'Bunch number [-]')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# Plot variantion of BBB offset over a few turns
from_i = 20
sim_i = 1

bbb_offsets = np.zeros(pos[:72, -from_i:, sim_i].shape)
xs = np.zeros(pos[:72, -from_i:, sim_i].shape)

print('Computing bunch-by-bunch offset...\n')
for i in range(bbb_offsets.shape[1]):
    bbb_offsets[:, i] = at.find_offset(pos[:72, -from_i + i, sim_i])
    xs[:, i] = np.linspace(0, len(bbb_offsets[:, i]), len(bbb_offsets[:, i]))


bbb_offsets = bbb_offsets * 1e9
xs = xs

if PLT_BUNCH_VAR_OVER_TURNS:
    m, ms = measured_offset()
    colormap = plt.cm.gist_ncar
    plt.figure()
    plt.title(f'Bunch-by-bunch Offset for $f_c =$ 200.1 MHz')
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, Ns))))
    plt.fill_between(xs[:,0], (m - ms) * 1e3, (m + ms) * 1e3,
                     color='b', alpha=0.3)
    plt.plot(xs[:,0], m * 1e3, linestyle='--', color='b', alpha=1, label='M')
    if MODE == 2:
        for i in range(bbb_offsets.shape[1]):
            plt.plot(xs[:,i], bbb_offsets[:,i] * 1e3, label=f'{ratio_array[i]}')
    else:
        for i in range(bbb_offsets.shape[1]):
            plt.plot(xs[:,i], bbb_offsets[:,i] * 1e3)
    plt.ylabel(f'$\Delta t$ [ps]')
    plt.xlabel(f'Bunch number [-]')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))



plt.show()