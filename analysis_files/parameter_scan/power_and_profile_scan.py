'''
File to plot both the power and the bunch-by-bunch offset from the parameter scans.

Author: Birk Emil Karlsen-Bæck
'''


# Imports ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os

# Utility files
import utility_files.data_utilities as dut
import utility_files.analysis_tools as at

# Functions from related files
from power_scan import get_power
from profile_scan import measured_offset

plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 16
    })

# Parameters ------------------------------------------------------------------
FREQ_CONFIG = 1
EXTENDED = False
VOLT_ERR = False
voltage_error = 1
CAV_TYPE = 3
MODE = 1                    # MODE 1 is transmitter gain, MODE 2 is LLRF
omit_ind = 10
shift_P = 0.11                # [%]


# Directories -----------------------------------------------------------------
mst_dir = os.getcwd()[:-len('analysis_files/parameter_scan')]

if MODE == 1:
    pwr_data_folder = f'power_scan_tr_fr{FREQ_CONFIG}/'
    prof_data_folder = f'profile_scan_tr_fr{FREQ_CONFIG}/'
    if VOLT_ERR:
        pwr_data_folder = f'power_scan_ve{100 * voltage_error:.0f}_tr_fr{FREQ_CONFIG}/'
        prof_data_folder = f'profile_scan_ve{100 * voltage_error:.0f}_tr_fr{FREQ_CONFIG}/'
else:
    pwr_data_folder = f'power_scan_llrf_fr{FREQ_CONFIG}/'
    prof_data_folder = f'profile_scan_llrf_fr{FREQ_CONFIG}/'
    if VOLT_ERR:
        pwr_data_folder = f'power_scan_ve{100 * voltage_error:.0f}_llrf_fr{FREQ_CONFIG}/'
        prof_data_folder = f'profile_scan_ve{100 * voltage_error:.0f}_llrf_fr{FREQ_CONFIG}/'

pwr_data_dir = mst_dir + 'data_files/' + pwr_data_folder
prof_data_dir = mst_dir + 'data_files/' + prof_data_folder

if EXTENDED:
    ratio_array = np.linspace(0.8, 1.2, 10) * 100
else:
    ratio_array = np.linspace(0.9, 1.1, 10) * 100

if MODE == 1:
    Ns = 10
else:
    ratio_array = np.array([5, 10, 14, 16, 20])
    Ns = 5

# Get power data --------------------------------------------------------------
if MODE == 1:
    sample_data = np.load(pwr_data_dir + f'{CAV_TYPE}sec_power_29999_tr{ratio_array[0]:.0f}.npy')
else:
    sample_data = np.load(pwr_data_dir + f'{CAV_TYPE}sec_power_29000_llrf{ratio_array[0]:.0f}.npy')

power = np.zeros((sample_data.shape[0], len(ratio_array)))

print('Fetching power data...\n')
for i in range(len(ratio_array)):
    for file in os.listdir(pwr_data_dir[:-1]):
        if MODE == 1:
            if file.endswith(f'tr{ratio_array[i]:.0f}.npy') and file.startswith(f'{CAV_TYPE}'):
                power[:,i] = np.load(pwr_data_dir + file)
        else:
            if file.endswith(f'llrf{ratio_array[i]:.0f}.npy') and file.startswith(f'{CAV_TYPE}'):
                power[:,i] = np.load(pwr_data_dir + file)

# Get bunch-by-bunch offset data ----------------------------------------------
if MODE == 1:
    sample_data = np.load(prof_data_dir + f'profile_29999_tr{ratio_array[0]:.0f}.npy')
else:
    sample_data = np.load(prof_data_dir + f'profile_29000_llrf{ratio_array[0]:.0f}.npy')

profiles = np.zeros((sample_data.shape[0], len(ratio_array)))
bins = np.zeros((sample_data.shape[0], len(ratio_array), ))

print('Fetching profiles...\n')
for i in range(len(ratio_array)):
    for file in os.listdir(prof_data_dir[:-1]):
        if MODE == 1:
            if file.endswith(f'tr{ratio_array[i]:.0f}.npy'):
                sample_i = np.load(prof_data_dir + file)
                profiles[:, i] = sample_i[:, 0]
                bins[:, i] = sample_i[:, 1]
        else:
            if file.endswith(f'llrf{ratio_array[i]:.0f}.npy'):
                sample_i = np.load(prof_data_dir + file)
                profiles[:, i] = sample_i[:, 0]
                bins[:, i] = sample_i[:, 1]

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


bbb_offsets = bbb_offsets[:,:omit_ind]
xs = xs[:,:omit_ind]

# Make the plot ---------------------------------------------------------------

# Power arrays
dt = 8e-9
t = np.linspace(0, dt * 65536, 65536)
ts = np.linspace(0, 4.990159369074305e-09 * 4620, 4620) + ((5.981e-6 - 4.983e-6) + (13.643e-6 - 12.829e-6)) / 2
t_s = 1e6
P_s = 1e-3
sec3_mean_tot, sec3_std_tot, sec4_mean_tot, sec4_std_tot = get_power()

# Bunch-by-bunch offset arrays
m, ms = measured_offset()


# Making the actual plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

cmap = plt.get_cmap('jet')
colors = cmap(np.linspace(0, 1, Ns))
fig.suptitle(f'$f_r =$ 200.038 MHz, $V =$ 6.7 MV')

# Power plot
ax[1].set_title('Power, 3-section')
ax[1].set_ylabel(r'Power [kW]')
ax[1].set_xlabel(r'$\Delta t$ [$\mu$s]')

if CAV_TYPE == 3:
    pass
    ax[1].plot(t * t_s, sec3_mean_tot * P_s, color='b', linestyle='--', label='M')
    # plt.plot(t * t_s, sec3_mean_tot * P_s - shift_P * sec3_mean_tot * P_s, color='b', linestyle='--', label='M')
    ax[1].fill_between(t * t_s, (sec3_mean_tot * 0.80) * P_s, (sec3_mean_tot * 1.20) * P_s, alpha=0.3, color='b')
else:
    pass
    ax[1].plot(t * t_s, sec4_mean_tot * P_s, color='b', linestyle='--', label='M')
    # plt.plot(t * t_s, sec4_mean_tot * P_s - shift_P * sec4_mean_tot * P_s, color='b', linestyle='--', label='M')
    ax[1].fill_between(t * t_s, (sec4_mean_tot * 0.80) * P_s, (sec4_mean_tot * 1.20) * P_s, alpha=0.3, color='b')

for i in range(len(ratio_array) - (len(ratio_array) - omit_ind)):
    ax[1].plot(ts * t_s, power[:, i] * P_s, label=f'{ratio_array[i]}', color=colors[i])

ax[1].set_xlim((3.25e-6 * t_s, 7.8e-6 * t_s))

# Bunch-by-bunch offset plot
ax[0].set_title('Bunch-by-bunch offset')
ax[0].set_ylabel(r'$\Delta t$ [ps]')
ax[0].set_xlabel(r'Bunch Number [-]')

cmap = plt.get_cmap('jet')
colors = cmap(np.linspace(0, 1, Ns))

ax[0].fill_between(xs[:,0], (m - ms) * 1e3, (m + ms) * 1e3, color='b', alpha=0.3)
ax[0].plot(xs[:,0], m * 1e3, linestyle='--', color='b', alpha=1, label='M')

for i in range(len(ratio_array) - omit_ind):
    ax[0].plot(xs[:, i], bbb_offsets[:, i] * 1e3, label=f'{ratio_array[i]}', color=colors[i])



handles, labels = ax[1].get_legend_handles_labels()
#fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.01, 0.5))


plt.show()


