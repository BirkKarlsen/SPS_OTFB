"""
File to make a figure of how the beam parameters where extracted from the beam profiles.

Author: Birk Emil Karlsen-Bæck
"""

# Imports -------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import utility_files.data_utilities as dut


plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.sans-serif': 'Computer Modern Sans Serif',
        'font.size': 16
    })

# Options -------------------------------------------------------------------------------------------------------------
PLT_BUNCH = True
PLT_FWHM = True
HOME = False
n_points = 60
turn_points = 99999
n_turns = 100
sample_rate = 10e9
sample_period = 1 / sample_rate

# Functions -----------------------------------------------------------------------------------------------------------
if HOME:
    pass
else:
    meas_dir = f'/Users/bkarlsen/cernbox/SPSOTFB_benchmark_data/data/2021-11-05/profiles_SPS_OTFB_flattop/'

profile_meas = f'MD_104'

profiles_data, profiles_data_corr = dut.import_profiles(meas_dir, [profile_meas])
profile_data_corr = profiles_data_corr[0, :]
profile_data_corr = profile_data_corr.reshape((n_turns, turn_points))

profile = profile_data_corr[0,:]
time_array = np.linspace(0, (turn_points - 1) * sample_period, turn_points)


if PLT_BUNCH:
    plt.figure()
    #plt.title(r'Measured Bunch')

    plt.plot(time_array, profile)



plt.show()
