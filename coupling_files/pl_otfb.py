'''
File to do preliminary simulations for coupling between the phase loop and OTFB.

Author: Birk Emil Karlsen-Bæck
'''

# Imports ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions import bigaussian
from blond.llrf.cavity_feedback import SPSCavityFeedback, CavityFeedbackCommissioning
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.llrf.beam_feedback import BeamFeedback

# Parameters ------------------------------------------------------------------
C = 2 * np.pi * 1100.009                        # Ring circumference [m]
gamma_t = 18.0                                  # Transition Gamma [-]
alpha = 1 / (gamma_t**2)                        # Momentum compaction factor [-]
p_s = 440e9                                     # Synchronous momentum [eV]
h = 4620                                        # 200 MHz harmonic number [-]
V = 10e6                                        # 200 MHz RF voltage [V]
phi = 0                                         # 200 MHz phase [-]

# Parameters for SPS OTFB
V_part = None
a_comb = 63/64
G_ff = 0.7
G_llrf = 20
G_tx = [1,
        1]
df = [0.18433333e6,
      0.2275e6]


# Parameters for SPS Phase Loop

# Parameters for SPS Synchronization Loop

# Parameters for the Simulation
N_m = int(5e5)                                  # Number of macro-particles for tracking
N_t = 4000                                      # Number of turns to track
total_intensity = 3385.8196e10                  # Total intensity

lxdir = '../'
N_bunches = 1


# Objects ---------------------------------------------------------------------

# SPS Ring
ring = Ring(C, alpha, p_s, Proton(), n_turns=N_t)


# RF Station
rfstation = RFStation(ring, [h], [V], [0], n_rf=1)


# Beam
bunch_intensities = np.load(lxdir + 'data_files/beam_parameters/avg_bunch_intensities_red.npy')
bunch_intensities = total_intensity * bunch_intensities / np.sum(bunch_intensities)  # normalize to 3385.8196 * 10**10
n_macro = N_m * N_bunches * bunch_intensities / np.sum(bunch_intensities)
beam = Beam(ring, int(np.sum(n_macro[:N_bunches])), int(total_intensity))


# Profile
profile = Profile(beam, CutOptions = CutOptions(cut_left=rfstation.t_rf[0,0] * (1000 - 2.5),
    cut_right=rfstation.t_rf[0,0] * (1000 + 72 * 5 + 125),
    n_slices=int(round(2**7 * (2.5 + 72 * 5 + 125)))))


# SPS Cavity Feedback
Commissioning = CavityFeedbackCommissioning(open_FF=True, debug=False, rot_IQ=1)
OTFB = SPSCavityFeedback(rfstation, beam, profile, post_LS2=True, V_part=V_part,
                         Commissioning=Commissioning, G_tx=G_tx, a_comb=a_comb,
                         G_llrf=G_llrf, df=df, G_ff=G_ff)

# SPS Phase-Loop
config = {'machine' : 'SPS_RL',
          'PL_gain' : 5000,
          'RL_gain' : 10,
          'sample_dE' : 1}
BFB = BeamFeedback(ring, rfstation, profile, configuration=config)


# Tracker Object without SPS OTFB
SPS_rf_tracker = RingAndRFTracker(rfstation, beam, CavityFeedback=None, Profile=profile,
                                  interpolation=True, BeamFeedback=BFB)
SPS_tracker = FullRingAndRF([SPS_rf_tracker])

if N_bunches != 1:
    beam.dE = np.load(
            lxdir + f'data_files/with_impedance/generated_beams/generated_beam_fwhm_{N_bunches}_dE_r.npy')
    beam.dt = np.load(
            lxdir + f'data_files/with_impedance/generated_beams/generated_beam_fwhm_{N_bunches}_dt_r.npy') + 1e-9
else:
    bigaussian(ring, rfstation, beam, 1.2e-9/4)
    beam.dt += 1000 * rfstation.t_rf[0,0] + 1e-9

profile.track()

dt_plot = 10
dt_track = 10

phase = np.zeros(N_t)

for i in range(N_t):
    #OTFB.track()
    #BFB.track()
    SPS_tracker.track()
    profile.track()
    phase[i] = SPS_rf_tracker.phi_rf[0, i]
    if i % dt_track == 0:
        print(f'Turn {i}')

    if i % dt_plot == 0:
        plt.figure()
        plt.plot(profile.bin_centers, profile.n_macroparticles * 1e5)
        plt.plot(profile.bin_centers, SPS_rf_tracker.total_voltage)
        plt.xlim((4.99e-6, 4.996e-6))
        plt.figure()
        plt.plot(phase[:i])


        plt.show()




