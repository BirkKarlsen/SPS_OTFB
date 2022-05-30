'''
File to compare power across different transmitter gain values.

Author: Birk Emil Karlsen-Bæck
'''

# Imports ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
from analysis_files.measurement_analysis.import_data import get_power

plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 16
    })


# Options ---------------------------------------------------------------------
FREQ_CONFIG = 3
EXTENDED = False
CAV_TYPE = 3
MODE = 1                    # MODE 1 is transmitter gain, MODE 2 is LLRF
omit_ind = 10
shift_P = 0.11                # [%]

# Plots
PLT_POWER = True
PLT_POWER_PTP = False

# Directories -----------------------------------------------------------------
mst_dir = os.getcwd()[:-len('analysis_files/parameter_scan')]

if MODE == 1:
    data_folder = f'tx_scan_fr{FREQ_CONFIG}_vc1_ve100_bl100_g20/'
else:
    data_folder = f'power_scan_llrf_fr{FREQ_CONFIG}/'
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
    sample_data = np.load(data_dir + f'{CAV_TYPE}sec_power_30000_tr{ratio_array[0]:.0f}.npy')
else:
    sample_data = np.load(data_dir + f'{CAV_TYPE}sec_power_29000_llrf{ratio_array[0]:.0f}.npy')

power = np.zeros((sample_data.shape[0], len(ratio_array)))

print('Fetching profiles...\n')
for i in range(len(ratio_array)):
    for file in os.listdir(data_dir[:-1]):
        if MODE == 1:
            if file.endswith(f'tr{ratio_array[i]:.0f}.npy') and file.startswith(f'{CAV_TYPE}'):
                power[:,i] = np.load(data_dir + file)
        else:
            if file.endswith(f'llrf{ratio_array[i]:.0f}.npy') and file.startswith(f'{CAV_TYPE}'):
                power[:,i] = np.load(data_dir + file)

if PLT_POWER:
    dt = 8e-9
    t = np.linspace(0, dt * 65536, 65536)
    ts = np.linspace(0, 4.990159369074305e-09 * 4620, 4620) + ((5.981e-6 - 4.983e-6) + (13.643e-6 - 12.829e-6)) / 2
    t_s = 1e6
    P_s = 1e-3

    sec3_mean_tot, sec3_std_tot, sec4_mean_tot, sec4_std_tot = get_power()

    plt.figure()

    plt.title('3-section Power for $f_c =$ 200.1 MHz')

    if CAV_TYPE == 3:
        pass
        plt.plot(t * t_s, sec3_mean_tot * P_s, color='b', linestyle='--', label='M')
        #plt.plot(t * t_s, sec3_mean_tot * P_s - shift_P * sec3_mean_tot * P_s, color='b', linestyle='--', label='M')
        plt.fill_between(t * t_s, (sec3_mean_tot * 0.80) * P_s, (sec3_mean_tot * 1.20) * P_s, alpha=0.3, color='b')
    else:
        pass
        plt.plot(t * t_s, sec4_mean_tot * P_s, color='b', linestyle='--', label='M')
        #plt.plot(t * t_s, sec4_mean_tot * P_s - shift_P * sec4_mean_tot * P_s, color='b', linestyle='--', label='M')
        plt.fill_between(t * t_s, (sec4_mean_tot * 0.80) * P_s, (sec4_mean_tot * 1.20) * P_s, alpha=0.3, color='b')
    colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, Ns))))

    for i in range(len(ratio_array) - (len(ratio_array) - omit_ind)):
        plt.plot(ts * t_s, power[:,i] * P_s, label=f'{ratio_array[i]}')

    plt.xlim((3.25e-6 * t_s, 1.65e-5 * t_s))
    plt.ylabel(r'Power [kW]')
    plt.xlabel(r'$\Delta t$ [$\mu$s]')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))



plt.show()