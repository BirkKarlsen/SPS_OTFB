'''
File to input real signals from the real SPS OTFB and compare the resulting signals with the measured
results.

author: Birk Emil Karlsen-Bæck
'''

# Import ----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import utility_files.analysis_tools as at
from scipy.interpolate import interp1d

from blond.llrf.cavity_feedback import SPSOneTurnFeedback, CavityFeedbackCommissioning

from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions import bigaussian


# Importing and processing the measured signals -------------------------------
antenna_abs, antenna_phase, antenna_t = at.import_measurement_signals(1, 20211106, 103331, 'Vcav')
antenna_IQ = antenna_abs * (np.cos(antenna_phase) + 1j * np.sin(antenna_phase))

turn_ind = at.find_closes_value(antenna_t, 23e-6)

antenna_IQ = antenna_IQ[:turn_ind]
antenna_t = antenna_t[:turn_ind]


# Resampeling the signal my interpolation of the signal
h = 4620
coarse_t = np.linspace(0, 23e-6, h)

ant_f = interp1d(antenna_t, antenna_IQ, fill_value='extrapolate')

antenna_coarse = ant_f(coarse_t)

plt.polar(np.angle(antenna_coarse), np.abs(antenna_coarse))
plt.show()

# Inputting the signal into the model

# Initialize a dummy simulation:
C = 2 * np.pi * 1100.009                        # Ring circumference [m]
gamma_t = 18.0                                  # Transition Gamma [-]
alpha = 1 / (gamma_t**2)                        # Momentum compaction factor [-]
p_s = 440e9                                     # Synchronous momentum [eV]
h = 4620                                        # 200 MHz harmonic number [-]
V = (0.911535 * 4 + 1.526871 * 2) * 1e6         # 200 MHz RF voltage [V]
phi = 0                                         # 200 MHz phase [-]

# Parameters for the Simulation
N_p = 1.15e11                                   # Number of particles per bunch
N_m = int(5e5)                                  # Number of macro-particles for tracking
N_t = 1000                                      # Number of turns to track
N_pre_track = 1000

# Ring
SPS_ring = Ring(C, alpha, p_s, Proton(), N_t)
rfstation = RFStation(SPS_ring, [h], [V], [0], n_rf=1)
beam = Beam(SPS_ring, N_m, N_p)
profile = Profile(beam, CutOptions=CutOptions(cut_left=rfstation.t_rf[0,0] * (-2.5),
    cut_right=rfstation.t_rf[0,0] * 2.5,
    n_slices=int(round(2**7 * 5))))

Commissioning = CavityFeedbackCommissioning(open_FF=True, debug=False, rot_IQ=1)
OTFB = SPSOneTurnFeedback(rfstation, beam, profile, n_sections=3, n_cavities=1,
                          G_llrf=20, a_comb=31/32, V_part=0.911535e6 / V)

for i in range(N_pre_track):
    OTFB.track_no_beam()

# Input the measured antenna voltage
OTFB.V_ANT[-h:] = antenna_IQ

# Go through the LLRF model:
OTFB.set_point()
OTFB.error_and_gain()
OTFB.comb()
OTFB.one_turn_delay()

