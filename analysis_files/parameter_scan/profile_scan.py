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
FREQ_CONFIG = 3
EXTENDED = False
MODE = 2

# Plots
PLT_PROFILE = True
PLT_BBB = True

# Directories -----------------------------------------------------------------
mst_dir = os.getcwd()[:-len('analysis_files/parameter_scan')]

if MODE == 1:
    data_folder = f'profile_scan_fr{FREQ_CONFIG}/'
else:
    data_folder = f'profile_scan_llrf_fr{FREQ_CONFIG}/'
data_dir = mst_dir + 'data_files/' + data_folder

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
    sample_data = np.load(data_dir + f'profile_29000_tr{ratio_array[0]:.0f}.npy')
else:
    sample_data = np.load(data_dir + f'profile_29000_llrf{ratio_array[0]:.0f}.npy')

profiles = np.zeros((sample_data.shape[0], len(ratio_array)))
bins = np.zeros((sample_data.shape[0], len(ratio_array), ))

print('Fetching profiles...\n')
for i in range(len(ratio_array)):
    for file in os.listdir(data_dir[:-1]):
        if MODE == 1:
            if file.endswith(f'tr{ratio_array[i]:.0f}.npy'):
                sample_i = np.load(data_dir + file)
                profiles[:, i] = sample_i[:, 0]
                bins[:, i] = sample_i[:, 1]
        else:
            if file.endswith(f'llrf{ratio_array[i]:.0f}.npy'):
                sample_i = np.load(data_dir + file)
                profiles[:, i] = sample_i[:, 0]
                bins[:, i] = sample_i[:, 1]

if PLT_PROFILE:
    plt.figure()
    colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, Ns))))
    plt.plot(bins, profiles)


# Find bunch-by-bunch offset --------------------------------------------------
print('Finding bunch positions...\n')
N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit, \
    Bunch_peaksFit, Bunch_Exponent, Goodness_of_fit, x_71, y_71 \
        = dut.getBeamPattern_3(bins[:,0], profiles,
                           distance=2**7 * 3, fit_option='fwhm', heightFactor=50,
                           wind_len=5, save_72_fits=False)


bbb_offsets = np.zeros(Bunch_positionsFit[:,:72].T.shape)
xs = np.zeros(Bunch_positionsFit[:,:72].T.shape)

print('Computing bunch-by-bunch offset...\n')
for i in range(len(ratio_array)):
    bbb_offsets[:, i] = at.find_offset(Bunch_positionsFit[i,:72])
    xs[:, i] = np.linspace(0, len(Bunch_positionsFit[0,:72]), len(Bunch_positionsFit[0,:72]))

omit_ind = 10
bbb_offsets = bbb_offsets[:,:omit_ind]
xs = xs[:,:omit_ind]


if PLT_BBB:
    m, ms = measured_offset()
    colormap = plt.cm.gist_ncar
    plt.figure()
    plt.title(f'200.1 MHz scan')
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, Ns))))
    plt.fill_between(xs[:,0], (m - ms) * 1e3, (m + ms) * 1e3,
                     color='b', alpha=0.3)
    plt.plot(xs[:,0], m * 1e3, linestyle='--', color='b', alpha=1, label='M')
    for i in range(len(ratio_array)):
        plt.plot(xs[:,i], bbb_offsets[:,i] * 1e3, label=f'{ratio_array[i]}')
    plt.legend()
    plt.ylabel(f'$\Delta t$ [ps]')
    plt.xlabel(f'Bunch number [-]')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))









plt.show()