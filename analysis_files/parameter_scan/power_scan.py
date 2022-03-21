'''
File to compare power across different transmitter gain values.

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
CAV_TYPE = 4

# Plots
PLT_POWER = True
PLT_POWER_PTP = False

# Directories -----------------------------------------------------------------
mst_dir = os.getcwd()[:-len('analysis_files/parameter_scan')]

data_folder = f'power_scan_fr{FREQ_CONFIG}/'
data_dir = mst_dir + 'data_files/' + data_folder

if EXTENDED:
    ratio_array = np.linspace(0.8, 1.2, 10)
else:
    ratio_array = np.linspace(0.9, 1.1, 10)


# Get data --------------------------------------------------------------------
sample_data = np.load(data_dir + f'{CAV_TYPE}sec_power_29999_tr{100 * ratio_array[0]:.0f}.npy')

power = np.zeros((sample_data.shape[0], len(ratio_array)))

print('Fetching profiles...\n')
for i in range(len(ratio_array)):
    for file in os.listdir(data_dir[:-1]):
        if file.endswith(f'tr{100 * ratio_array[i]:.0f}.npy') and file.startswith(f'{CAV_TYPE}'):
            power[:,i] = np.load(data_dir + file)

if PLT_POWER:
    plt.figure()
    colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, 10))))
    plt.plot(power)


plt.show()