'''
This file does a pretracking of the SPS OTFB to see if it will be consistent after the sign-change.

author: Birk Emil Karlsen-Bæck
'''


# Imports -------------------------------------------------------------------------------------------------------------

print('Importing...\n')
import matplotlib.pyplot as plt
import numpy as np
import utility_files.data_utilities as dut
import os.path
from datetime import date
import utility_files.analysis_tools as at

from blond.llrf.cavity_feedback import SPSOneTurnFeedback, CavityFeedbackCommissioning
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions_multibunch import matched_from_distribution_density_multibunch
from blond.trackers.tracker import FullRingAndRF, RingAndRFTracker
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from blond.impedances.impedance_sources import InputTable

plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 16
    })


# Parameters ----------------------------------------------------------------------------------------------------------
C = 2 * np.pi * 1100.009                        # Ring circumference [m]
gamma_t = 18.0                                  # Transition Gamma [-]
alpha = 1 / (gamma_t**2)                        # Momentum compaction factor [-]
p_s = 440e9                                     # Synchronous momentum [eV]
h = 4620                                        # 200 MHz harmonic number [-]
V = (0.911535 * 4 + 1.526871 * 2) * 1e6         # 200 MHz RF voltage [V]
phi = 0                                         # 200 MHz phase [-]


# Parameters for the Simulation
N_m = int(5e5)                                  # Number of macro-particles for tracking
N_t = 1000                                      # Number of turns to track



# Ring
SPS_ring = Ring(C, alpha, p_s, Proton(), N_t)

# RFStation
rfstation = RFStation(SPS_ring, [h, 4 * h], [V, 0.19 * V], [0, np.pi], n_rf=2)


# SINGLE BUNCH FIRST
# Beam
total_intensity = 3385.8196 * 10**10



beam = Beam(SPS_ring, int(N_m * 288), int(total_intensity))

# Profile
profile = Profile(beam, CutOptions = CutOptions(cut_left=0.e-9,
    cut_right=rfstation.t_rev[0], n_slices=2**7 * 4620))

# One Turn Feedback
V_part = 0.5442095845867135
# TODO: Run with Gtx of 1

G_tx_ls = [1.0, 1.0]
#G_llrf_ls = [41.751786, 35.24865]
#llrf_g = G_llrf_ls


Commissioning = CavityFeedbackCommissioning(open_FF=True, debug=False,
                                            rot_IQ=1)
OTFB = SPSOneTurnFeedback(rfstation, beam, profile, 3, V_part=1,
                         Commissioning=Commissioning, G_tx=0.9, a_comb=31/32,
                         G_llrf=20)   # TODO: change back to only 20


for i in range(N_t):
    OTFB.track_no_beam()



at.plot_IQ(OTFB.V_ANT[-OTFB.n_coarse:],
           OTFB.V_IND_COARSE_GEN[-OTFB.n_coarse:],
           OTFB.V_IND_COARSE_BEAM[-OTFB.n_coarse:],
           wind=V * 1.2)

plt.show()