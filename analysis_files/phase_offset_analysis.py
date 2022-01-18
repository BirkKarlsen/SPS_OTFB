'''
File to analyse the bunch-by-bunch offset from simulations and from the measurements.

author: Birk Emil Karlsen-Bæck
'''

import utility_files.analysis_tools as at
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

t_rf = 4.99015937e-09

# Importing the profiles - Sim files are from with_impedance/Jan-02-2022/central_freq1/30000turns_fwhm_288/
sim_p = np.load('../data_files/profiles/generated_profile_fwhm_288_end_30000.npy')
sim_b = np.load('../data_files/profiles/generated_profile_bins_fwhm_288_end_30000.npy')
sim_p = sim_p / np.sum(sim_p)

mea_p = at.import_measured_profile('../data_files/profiles/MD_104.npy', 0)
mea_b = np.linspace(0, len(mea_p)*(1/10e9), len(mea_p))
mea_p = mea_p / np.sum(np.abs(mea_p)) * 0.7

# Finding positions of the bunches
sim_pos = at.positions_simulated(sim_p, sim_b) * 1e-9
mea_pos = at.positions_measured(mea_p, mea_b) * 1e-9

# Moving to approximately same place in time
dt = (np.mean(sim_pos[:72]) - np.mean(mea_pos[:72]))
mea_b = mea_b + dt

# Finding new positions
sim_pos = at.positions_simulated(sim_p, sim_b) * 1e-9
mea_pos = at.positions_measured(mea_p, mea_b) * 1e-9

# Finding bunch-by-bunch offsets
mea_bbb, sim_bbb = at.find_offset_fit(mea_pos[:72], sim_pos[:72])
mea_bbb_trf, sim_bbb_trf = at.find_offset_trf(mea_pos[:72], sim_pos[:72], t_rf)

plt.figure()
plt.plot(sim_bbb, color='r', label='Sim')
plt.plot(mea_bbb, color='b', label='Meas')
plt.legend()

plt.figure()
plt.plot(sim_b, sim_p, color='r', label='Sim')
plt.plot(mea_b, mea_p, color='b', label='Meas')
plt.legend()

#plt.figure()
#plt.plot(mea_bbb_trf)
#plt.plot(sim_bbb_trf)
plt.show()