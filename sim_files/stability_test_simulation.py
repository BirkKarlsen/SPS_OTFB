'''
File to simulate a simple uniform case with the OTFB to check for stability

Author: Birk Emil Karlsen-Bæck
'''

# Options -------------------------------------------------------------------------------------------------------------
LXPLUS = True
mstdir = 'simple_stability/'
fit_type = 'fwhm'
dt_track = 1000
dt_ptrack = 10
dt_plot = 1000
dt_save = 1000
tr = 1
bunch_length_factor = 1.0

# Imports -------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os.path
from datetime import date

import utility_files.analysis_tools as at
import utility_files.data_utilities as dut

from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions import bigaussian
from blond.llrf.cavity_feedback import SPSCavityFeedback, CavityFeedbackCommissioning
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF

# Parameters ----------------------------------------------------------------------------------------------------------
C = 2 * np.pi * 1100.009                        # Ring circumference [m]
gamma_t = 18.0                                  # Transition Gamma [-]
alpha = 1 / (gamma_t**2)                        # Momentum compaction factor [-]
p_s = 440e9                                     # Synchronous momentum [eV]
h = 4620                                        # 200 MHz harmonic number [-]
V = (0.911535 * 4 + 1.526871 * 2) * 1e6         # 200 MHz RF voltage [V]
phi = 0                                         # 200 MHz phase [-]

# OTFB Parameters
V_part = 0.5442095845867135                     # Voltage partitioning [-]
G_tx = [0.3/0.33,
        0.3/0.33]
G_llrf = 20
df = [0,
      0]
G_ff = 1

# Parameters for the Simulation
N_m = int(5e5)                                  # Number of macro-particles for tracking
N_t = 30000                                      # Number of turns to track
bnh_spc = 5
bth_spc = 50
total_intensity = 3385.8196e10

# LXPLUS Simulation Configurations ------------------------------------------------------------------------------------
if LXPLUS:
    lxdir = "/afs/cern.ch/work/b/bkarlsen/Simulation_Files/SPS_OTFB/"
else:
    lxdir = '../'
N_bunches = 288


# Objects -------------------------------------------------------------------------------------------------------------
# SPS Ring
ring = Ring(C, alpha, p_s, Proton(), n_turns=N_t)

# RF Station
rfstation = RFStation(ring, [h], [V], [0], n_rf=1)

# Beam
beam_single_bunch = Beam(ring, int(N_m), int(total_intensity / N_bunches))
bigaussian(ring, rfstation, beam_single_bunch, sigma_dt=1.2e-9/4, seed=1234)

beam = Beam(ring, int(N_bunches * N_m), int(total_intensity))
j = 0
batch_i = 0
for i in range(N_bunches):
    beam.dE[i * N_m: (i + 1) * N_m] = beam_single_bunch.dE
    beam.dt[i * N_m: (i + 1) * N_m] = beam_single_bunch.dt + (bnh_spc * (j - batch_i) + batch_i * bth_spc) * rfstation.t_rf[0,0]
    j += 1
    if j % 72 == 0:
        batch_i += 1


# Profile
profile = Profile(beam, CutOptions=CutOptions(cut_left=rfstation.t_rf[0, 0] * (-2.5),
            cut_right=rfstation.t_rf[0, 0] * (72 * 5 * 4 + 250 * 3 + 125),
            n_slices=int(round(2 ** 7 * (2.5 + 72 * 5 * 4 + 250 * 3 + 125)))))

profile.track()

# SPS Cavity Controller
Commissioning = CavityFeedbackCommissioning(open_FF=True, debug=False, rot_IQ=1)
OTFB = SPSCavityFeedback(rfstation, beam, profile, post_LS2=True, V_part=V_part,
                             Commissioning=Commissioning, G_tx=G_tx, a_comb=31/32,
                             G_llrf=G_llrf, df=df, G_ff=G_ff)


# Tracker Object with SPS Cavity Feedback -----------------------------------------------------------------------------
SPS_rf_tracker = RingAndRFTracker(rfstation, beam, CavityFeedback=OTFB, Profile=profile,
                                  interpolation=True)
SPS_tracker = FullRingAndRF([SPS_rf_tracker])

# Set up directories for saving results -------------------------------------------------------------------------------
today = date.today()
sim_dir = f'data_files/with_impedance/{today.strftime("%b-%d-%Y")}/{mstdir}{N_t}turns_fwhm_{N_bunches}/'
sdir = lxdir + sim_dir + f'sim_data/'

np.save(lxdir + f'data_files/with_impedance/profile_data/generated_profile_{fit_type}_{N_bunches}_r', profile.n_macroparticles)
np.save(lxdir + f'data_files/with_impedance/profile_data/generated_profile_bins_{fit_type}_{N_bunches}_r', profile.bin_centers)
if not os.path.exists(lxdir + sim_dir + f'fig/'):
    os.makedirs(lxdir + sim_dir + f'fig/')

if not os.path.exists(sdir):
    os.makedirs(lxdir + sim_dir + f'sim_data/')

# Show vital parameters for the simulation before tracking --------------------
print('----- Simulation Infomation -----')
print('Voltage:')
print('\tV =', V)
print('\tV_part', V_part)
print('One-Turn Feedback:')
print('\ta_comb =', OTFB.OTFB_1.a_comb, OTFB.OTFB_2.a_comb)
print('\tG_llrf =', OTFB.OTFB_1.G_llrf, OTFB.OTFB_2.G_llrf)

# Particle tracking -----------------------------------------------------------

fwhm_arr = np.zeros((N_bunches, N_t//dt_ptrack))
pos_arr = np.zeros((N_bunches, N_t//dt_ptrack))
pos_fit_arr = np.zeros((N_bunches, N_t//dt_ptrack))
max_pow_arr = np.zeros((2, N_t//dt_ptrack))
max_V_arr = np.zeros((2, N_t//dt_ptrack))
int_arr = np.zeros(N_t//dt_ptrack)
n = 0

for i in range(N_t):
    SPS_tracker.track()
    profile.track()
    OTFB.track()

    #sig = OTFB.OTFB_1.I_FINE_BEAM[-profile.n_slices:]
    #plt.figure()
    #plt.plot(np.abs(sig) *
    #         np.sin(rfstation.omega_rf[0,0] * profile.bin_centers + np.angle(sig) + np.pi / 2))
    #plt.plot(profile.n_macroparticles * beam.ratio * 2 * 1.6e-19 / profile.bin_size)
    #plt.show()

    if i % dt_ptrack == 0:
        # Power
        OTFB.OTFB_1.calc_power()
        OTFB.OTFB_2.calc_power()

        fwhm_arr[:, n], pos_arr[:, n], pos_fit_arr[:, n], x_72, y_72 = dut.bunch_params(profile,
                                                                                        get_72=False)
        int_arr[n] = beam.intensity
        n += 1

    if i % dt_save == 0:
        dut.plot_params(fwhm_arr, pos_arr, pos_fit_arr,
                            max_pow_arr, max_V_arr, lxdir + sim_dir,
                            rfstation.t_rf[0,0], i, n - 1,
                            MB=True)

        dut.save_params(fwhm_arr, pos_arr, pos_fit_arr,
                            max_pow_arr, max_V_arr, lxdir + sim_dir)

        dut.plot_ramp(int_arr, i, n - 1, lxdir + sim_dir)


    if i % dt_plot == 0:

        OTFB.OTFB_1.calc_power()
        OTFB.OTFB_2.calc_power()

        dut.save_plots_OTFB(OTFB, lxdir + sim_dir + f'fig/', i)
        dut.save_data(OTFB, lxdir + sim_dir + f'sim_data/', i)

        dut.save_profile(profile, lxdir + sim_dir + f'sim_data/', i)
        dut.plot_bbb_offset(pos_fit_arr[:, n-1], 4, lxdir + sim_dir + f'fig/', i)


OTFB.OTFB_1.calc_power()
OTFB.OTFB_2.calc_power()
dut.save_plots_OTFB(OTFB, lxdir + sim_dir + f'fig/', N_t)

dut.plot_bbb_offset(pos_fit_arr[:, n - 1], 4, lxdir + sim_dir + f'fig/', N_t)

dut.plot_params(fwhm_arr, pos_arr, pos_fit_arr,
                    max_pow_arr, max_V_arr, lxdir + sim_dir,
                    rfstation.t_rf[0, 0], N_t, n - 1,
                    MB=True)
dut.save_params(fwhm_arr, pos_arr, pos_fit_arr,
                    max_pow_arr, max_V_arr, lxdir + sim_dir)

# Save the results to their respective directories
dut.save_data(OTFB, lxdir + sim_dir + f'sim_data/', N_t)

if not os.path.exists(lxdir + sim_dir + f'profile_data/'):
    os.makedirs(lxdir + sim_dir + f'profile_data/')
np.save(lxdir + sim_dir + f'profile_data/generated_profile_{fit_type}_{N_bunches}_end_{N_t}',
        profile.n_macroparticles)
np.save(lxdir + sim_dir + f'profile_data/generated_profile_bins_{fit_type}_{N_bunches}_end_{N_t}',
        profile.bin_centers)

