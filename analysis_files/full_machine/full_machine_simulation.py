'''
Simulation of a full SPS machine to compare with theoretical values.

Author: Birk Emil Karlsen-Bæck
'''

# Imports ---------------------------------------------------------------------
print('Importing...\n')
import numpy as np
import matplotlib.pyplot as plt

from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions import bigaussian
from blond.llrf.cavity_feedback import SPSCavityFeedback, CavityFeedbackCommissioning
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF

# Parameters ------------------------------------------------------------------
C = 2 * np.pi * 1100.009                        # Ring circumference [m]
gamma_t = 18.0                                  # Transition Gamma [-]
alpha = 1 / (gamma_t**2)                        # Momentum compaction factor [-]
p_s = 450e9                                     # Synchronous momentum [eV]
h = 4620                                        # 200 MHz harmonic number [-]
V = 10e6                                        # 200 MHz RF voltage [V]
phi = 0                                         # 200 MHz phase [-]

# Parameters for the Simulation
N_t = 1000                                      # Number of turns to track

# SPS Cavity Controler Parameters
df = [0.18433333e6,
      0.2275e6]
G_tx = [0.22908800255477224,
        0.4294032663036014]
G_llrf = 20
a_comb = 63/64

# Beam parameters
N_m = int(5e5)                                  # Number of macro-particles for tracking
N_p = 2.3e11                                    # Number of protons per bunch
spacing = 5                                     # Spacing between bunches in units of buckets
N_bunches = h // 5                              # Number of bunches
bl = 1.2e-9                                     # Bunch length [s]


# Objects ---------------------------------------------------------------------
print('Setting up...\n')

# SPS Ring
ring = Ring(C, alpha, p_s, Proton(), n_turns=N_t)


# RF Station
rfstation = RFStation(ring, [h], [V], [0], n_rf=1)


# Beam
beam_single = Beam(ring, int(N_m), int(N_p))
bigaussian(ring, rfstation, beam_single, sigma_dt=bl/4, seed=1234)

beam = Beam(ring, int(N_bunches * N_m), int(N_bunches * N_p))
for i in range(N_bunches):
    beam.dt[i * N_m:(i + 1) * N_m] = beam_single.dt + i * rfstation.t_rf[0, 0] * spacing
    beam.dE[i * N_m:(i + 1) * N_m] = beam_single.dE


# Profile
profile = Profile(beam, CutOptions = CutOptions(cut_left=0, cut_right=rfstation.t_rev[0],
    n_slices=int(round(2**7 * 4620))))


# SPS Cavity Feedback
Commissioning = CavityFeedbackCommissioning(open_FF=True, debug=False, rot_IQ=1)
OTFB = SPSCavityFeedback(rfstation, beam, profile, post_LS2=True, Commissioning=Commissioning,
                         G_tx=G_tx, a_comb=a_comb, G_llrf=G_llrf, df=df)


# Tracker Object without SPS OTFB
SPS_rf_tracker = RingAndRFTracker(rfstation, beam, CavityFeedback=OTFB,
                                  Profile=profile, interpolation=True)
SPS_tracker = FullRingAndRF([SPS_rf_tracker])

profile.track()


# Tracking --------------------------------------------------------------------
print('Simulating...\n')

for i in range(N_t):
    # Assuming a static beam and therefore only tracking the OTFB
    OTFB.track()


