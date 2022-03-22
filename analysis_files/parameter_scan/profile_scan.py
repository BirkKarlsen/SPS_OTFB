'''
File to compare bunch-by-bunch offset across different transmitter gain values.

Author: Birk Emil Karlsen-Bæck
'''


# Imports ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import utility_files.data_utilities as dut
import utility_files.analysis_tools as at

plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 16
    })


# Options ---------------------------------------------------------------------
FREQ_CONFIG = 3
EXTENDED = False

# Plots
PLT_PROFILE = True
PLT_BBB = True

# Directories -----------------------------------------------------------------
mst_dir = os.getcwd()[:-len('analysis_files/parameter_scan')]

data_folder = f'profile_scan_fr{FREQ_CONFIG}/'
data_dir = mst_dir + 'data_files/' + data_folder

if EXTENDED:
    ratio_array = np.linspace(0.8, 1.2, 10)
else:
    ratio_array = np.linspace(0.9, 1.1, 10)


# Get data --------------------------------------------------------------------
sample_data = np.load(data_dir + f'profile_29000_tr{100 * ratio_array[0]:.0f}.npy')

profiles = np.zeros((sample_data.shape[0], len(ratio_array)))
bins = np.zeros((sample_data.shape[0], len(ratio_array), ))

print('Fetching profiles...\n')
for i in range(len(ratio_array)):
    for file in os.listdir(data_dir[:-1]):
        if file.endswith(f'tr{100 * ratio_array[i]:.0f}.npy'):
            sample_i = np.load(data_dir + file)
            profiles[:, i] = sample_i[:, 0]
            bins[:, i] = sample_i[:, 1]

if PLT_PROFILE:
    plt.figure()
    colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, 10))))
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
    colormap = plt.cm.gist_ncar
    plt.figure()
    plt.title(f'200.1 MHz scan')
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, 10))))
    plt.plot(xs, bbb_offsets)
    plt.ylabel(f'$\Delta t$ [ns]')
    plt.xlabel(f'Bunch number [-]')









plt.show()