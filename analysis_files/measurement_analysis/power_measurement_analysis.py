'''
File to analyse the power measurements performed november 2021.

Author: Birk Emil Karlsen-Bæck
'''


# Imports -------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os

import utility_files.data_utilities as dut
import utility_files.analysis_tools as at
from analysis_files.full_machine.full_machine_theoretical_estimates import theoretical_power


# Options -------------------------------------------------------------------------------------------------------------
PLT_POWER = True
CAVITY = 6
n_points = 65536

# Analysis ------------------------------------------------------------------------------------------------------------
# Values from Danilo
# mean
measured_voltages = np.array([0.9 * (1 + 0.03), 0.9 * (1 - 0.068), 1.2 * (1 - 0.208),
                              0.9 * (1 - 0.129), 0.9 * (1 - 0.168), 1.2 * (1 - 0.103)]) * 1e6

# error
measured_voltages_error = np.array([0.9 * 0.01, 0.9 * 0.003, 1.2 * 0.004,
                                    0.9 * 0.002, 0.9 * 0.007, 1.2 * 0.002]) * 1e6

# Directories
data_folder = '/Users/bkarlsen/cernbox/SPSOTFB_benchmark_data/data/2021-11-05/'
file_prefix = f'sps_otfb_data__all_buffers__cavity{CAVITY}__flattop__20211106_10'

# Retrieve data from CERNbox
file_names = dut.file_names_in_dir_from_prefix(data_folder, file_prefix)

power, time = at.retrieve_power(data_folder, file_names, CAVITY, n_points)
at.plot_power_measurement_shots(power, time)

plt.show()