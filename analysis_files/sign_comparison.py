r'''
File to perform a comparison between the sign-change in the OTFB model

author: Birk Emil Karlsen-Bæck
'''

import numpy as np
import matplotlib.pyplot as plt
import utility_files.analysis_tools as at

# Directories -----------------------------------------------------------------
data_dir = '../data_files/with_impedance/Jan-26-2022/'
p_dir = 'std_sign/100turns_fwhm_288/sim_data/'
m_dir = 'rev_sign_3/100turns_fwhm_288/sim_data/'


# Options for the file --------------------------------------------------------
start_turn = 2
end_turn = 98
PLOT_PROFILE = False
PLOT_IQ = True


# Profile related plots -------------------------------------------------------
p_profile, p_bin = at.import_profiles_from_turn(data_dir, p_dir, start_turn)
m_profile, m_bin = at.import_profiles_from_turn(data_dir, m_dir, start_turn)

if PLOT_PROFILE:
    at.plot_profiles(p_profile, p_bin, m_profile, m_bin)

p_profile, p_bin = at.import_profiles_from_turn(data_dir, p_dir, end_turn)
m_profile, m_bin = at.import_profiles_from_turn(data_dir, m_dir, end_turn)

if PLOT_PROFILE:
    at.plot_profiles(p_profile, p_bin, m_profile, m_bin)


# IQ-signal plots -------------------------------------------------------------
p_Va3, p_Vg3, p_Vb3, p_Va4, p_Vg4, p_Vb4 = at.import_OTFB_signals(data_dir, p_dir, start_turn)
m_Va3, m_Vg3, m_Vb3, m_Va4, m_Vg4, m_Vb4 = at.import_OTFB_signals(data_dir, m_dir, start_turn)

p_sigs = np.array([p_Va3, p_Vg3, p_Vb3]).T
m_sigs = np.array([m_Va3, m_Vg3, m_Vb3]).T

print(np.angle(-1))

print(p_sigs.shape)
if PLOT_IQ:
    at.plot_compare_IQs(p_sigs, m_sigs)


p_Va3, p_Vg3, p_Vb3, p_Va4, p_Vg4, p_Vb4 = at.import_OTFB_signals(data_dir, p_dir, end_turn)
m_Va3, m_Vg3, m_Vb3, m_Va4, m_Vg4, m_Vb4 = at.import_OTFB_signals(data_dir, m_dir, end_turn)


p_sigs = np.array([p_Va3, p_Vg3, p_Vb3]).T
m_sigs = np.array([m_Va3, m_Vg3, m_Vb3]).T

if PLOT_IQ:
    at.plot_compare_IQs(p_sigs, m_sigs)


plt.show()