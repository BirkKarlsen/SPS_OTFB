'''
Analysis of the cavity signals taken during the FF commissioning on 16th November 2022.

Author: Birk Emil Karlsen-Bæck
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt
import os

import beam_dynamics_tools.data_visualisation.make_plots_pretty
from beam_dynamics_tools.data_management.importing_data import find_files_in_folder_starting_and_ending_with

# Directories
fdir = f'../../data_files/ff_measurements/cavity_signals/'

# Options
PLT_SIGNALS = True

# Fetching data
filenames = find_files_in_folder_starting_and_ending_with(fdir, prefix='Acq')

sig = np.loadtxt(fdir + filenames[1])

plt.figure()
plt.title('Voltage')
plt.plot(sig[:, 0] / 1e6)
plt.plot(sig[:, 2] / 1e6)
plt.plot(sig[:, 4] / 1e6)
plt.plot(sig[:, 6] / 1e6)
plt.plot(sig[:, 8] / 1e6)
plt.plot(sig[:, 10] / 1e6)
plt.ylabel(r'$V_{rf}$ [MV]')

plt.figure()
plt.title('Power')
plt.plot(sig[:, 12] / 1e3)
plt.plot(sig[:, 14] / 1e3)
plt.plot(sig[:, 16] / 1e3)
plt.plot(sig[:, 18] / 1e3)
plt.plot(sig[:, 20] / 1e3)
plt.plot(sig[:, 22] / 1e3)
plt.ylabel(r'$P_{gen}$ [kW]')

plt.show()

