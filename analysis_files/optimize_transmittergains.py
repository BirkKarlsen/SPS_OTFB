'''
Python-script to optimize the transmitter-gains in the SPS

author: Birk Emil Karlsen-Bæck
'''
# Imports ---------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions import bigaussian
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.llrf.cavity_feedback import SPSCavityFeedback, CavityFeedbackCommissioning
from blond.trackers.tracker import RingAndRFTracker


# Parameters ------------------------------------------------------------------
C = 2 * np.pi * 1100.009                        # Ring circumference [m]
gamma_t = 18.0                                  # Transition Gamma [-]
alpha = 1 / (gamma_t**2)                        # Momentum compaction factor [-]
p_s = 440e9                                     # Synchronous momentum [eV]
h = 4620                                        # 200 MHz harmonic number [-]
V = 10e6                                        # 200 MHz RF voltage [V]
phi = 0                                         # 200 MHz phase [-]

N_m = int(5e5)                                  # Number of macro-particles for tracking
N_p = 1.15e11                                   # Number of particles
N_t = 1                                         # Number of turns to track
sigma_dt = 1.2e-9

#G_tx = [0.251402590786449, 0.511242728131293] For the old signs from before 02/02/2022 both at 200.222 MHz
df = [0.18433333e6, 0.2275e6]
G_tx = [0.25154340605790062590, 0.510893981556323] # For the inverted real axis with 200.222 MHz

df = [0, 0.2275e6]
G_tx = [0.2607509145194842, 0.510893981556323]
'''
For the PostLS2 scenario with both cavities at 200.222 MHz the optimized transmitter gains are
[0.251402590786449, 0.511242728131293]
'''

# Objects ---------------------------------------------------------------------

# SPS ring
ring = Ring(C, alpha, p_s, Proton(), N_t)

# RF-station
rfstation = RFStation(ring, [h], [V], [phi])

# Beam
beam = Beam(ring, N_m, N_p)

# Profile
profile = Profile(beam, CutOptions = CutOptions(cut_left=0.e-9,
    cut_right=rfstation.t_rev[0], n_slices=2**7 * 4620))

bigaussian(ring, rfstation, beam, sigma_dt)


# SPS cavity controller
Commissioning = CavityFeedbackCommissioning(open_FF=True, debug=False)

OTFB = SPSCavityFeedback(rfstation, beam, profile, post_LS2=True,
                         Commissioning=Commissioning, G_tx=G_tx, a_comb=63/64,
                         G_llrf=20, df=df)

# Comparison

target3 = 10 * 4 * 3 / (4 * 3 + 2 * 4)
target4 = 10 * 2 * 4 / (4 * 3 + 2 * 4)

ant3 = np.mean(np.abs(OTFB.OTFB_1.V_ANT[-h:])) / 1e6
ant4 = np.mean(np.abs(OTFB.OTFB_2.V_ANT[-h:])) / 1e6
print(f'Desired 3-section: {target3} MV')
print(f'Model 3-section: {ant3} MV')
print()
print(f'Desired 4-section: {target4} MV')
print(f'Model 4-section: {ant4} MV')

diff3 = np.abs(target3 - ant3) * 100 / target3
diff4 = np.abs(target4 - ant4) * 100 / target4
print()
print(f'3-section difference: {diff3} %')
print(f'4-section difference: {diff4} %')