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

# Simulations with no phase-shift
profile_ns, bin_ns = at.import_and_normalize_profile('../data_files/profiles_corrected/profile_4000.npy')

# Simulations with 180 degree phase-shift
#profile_rs, bin_rs = at.import_and_normalize_profile('../data_files/profile_comparison/profile_4000_reversed_sign.npy')

# Simulations with 180 degree phase-shift and inverse phase correction
#profile_rs_rp, bin_rs_rp = at.import_and_normalize_profile(
#    '../data_files/profile_comparison/profile_4000_reversed_sign_phi.npy')

# From measurements
mea_p = at.import_measured_profile('../data_files/profiles/MD_104.npy', 0)
mea_b = np.linspace(0, len(mea_p)*(1/10e9), len(mea_p))
mea_p = mea_p / np.sum(np.abs(mea_p)) * 0.7


# Finding positions of the bunches
pos_ns = at.positions_simulated(profile_ns, bin_ns) * 1e-9
#pos_rs = at.positions_simulated(profile_rs, bin_rs) * 1e-9
#pos_rs_rp = at.positions_simulated(profile_rs_rp, bin_rs_rp) * 1e-9
mea_pos = at.positions_measured(mea_p, mea_b) * 1e-9

# Moving to approximately same place in time
dt = (np.mean(pos_ns[:72]) - np.mean(mea_pos[:72]))
mea_b = mea_b + dt

# Finding new positions
pos_ns = at.positions_simulated(profile_ns, bin_ns) * 1e-9
#pos_rs = at.positions_simulated(profile_rs, bin_rs) * 1e-9
#pos_rs_rp = at.positions_simulated(profile_rs_rp, bin_rs_rp) * 1e-9
mea_pos = at.positions_measured(mea_p, mea_b) * 1e-9

# Finding bunch-by-bunch offsets
mea_bbb, bbb_ns = at.find_offset_fit(mea_pos[:72], pos_ns[:72])
#mea_bbbm, bbb_rs = at.find_offset_fit(mea_pos[:72], pos_rs[:72])
#mea_bbbm, bbb_rs_rp = at.find_offset_fit(mea_pos[:72], pos_rs_rp[:72])

plt.figure()
plt.plot(bbb_ns, color='r', label='Normal')
#plt.plot(bbb_rs, color='b', label='Reversed')
#plt.plot(bbb_rs_rp, color='g', label=r'Reversed $\phi$')
plt.plot(mea_bbb, color='black', label='Meas')
plt.legend()

plt.figure()
plt.plot(bin_ns, profile_ns, color='r', label='Normal')
#plt.plot(bin_rs, profile_rs, color='b', label='Reversed')
#plt.plot(bin_rs_rp, profile_rs_rp, color='g', label=r'Reversed $\phi$')
plt.plot(mea_b, mea_p, color='black', label='Meas')
plt.legend()

#plt.figure()
#plt.plot(mea_bbb_trf)
#plt.plot(sim_bbb_trf)
plt.show()