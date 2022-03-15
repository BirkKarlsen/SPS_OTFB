'''
Initial first simulation to test FF model.

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
from blond.beam.distributions_multibunch import matched_from_distribution_density_multibunch
from blond.llrf.cavity_feedback import SPSCavityFeedback, CavityFeedbackCommissioning
from blond.impedances.impedance import TotalInducedVoltage, InducedVoltageFreq
from blond.impedances.impedance_sources import InputTable
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF

# Parameters ------------------------------------------------------------------
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
dtrack = 50
dplot = 200
dprof = 200

# OTFB parameters
V_part = 0.5442095845867135                     # Voltage partitioning [-]
df = [0.18433333e6,
      0.2275e6]
G_tx = [0.229377820916177,
        0.430534529571209]
G_llrf = 16
a_comb = 31/32
G_ff = 0.7

N_bunches = 72
lxdir = '../'
total_intensity = 3385.8196 * 10**10 / 4
fit_type = 'fwhm'


# Objects ---------------------------------------------------------------------
print('Setting up...\n')

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
    cut_right=rfstation.t_rf[0,0] * (1000 + 72 * 5 * 4 + 250 * 3 + 125),
    n_slices=int(round(2**7 * (2.5 + 72 * 5 * 4 + 250 * 3 + 125)))))


# SPS Cavity Feedback
Commissioning = CavityFeedbackCommissioning(open_FF=False, debug=False, rot_IQ=1)
OTFB = SPSCavityFeedback(rfstation, beam, profile, post_LS2=True, V_part=V_part,
                         Commissioning=Commissioning, G_tx=G_tx, a_comb=31/32,
                         G_llrf=G_llrf, df=df, G_ff=G_ff)


# Tracker Object without SPS OTFB
SPS_rf_tracker = RingAndRFTracker(rfstation, beam, CavityFeedback=OTFB,
                                  Profile=profile, interpolation=True)
SPS_tracker = FullRingAndRF([SPS_rf_tracker])

beam.dE = np.load(
        lxdir + f'data_files/with_impedance/generated_beams/generated_beam_{fit_type}_{N_bunches}_dE_r.npy')
beam.dt = np.load(
        lxdir + f'data_files/with_impedance/generated_beams/generated_beam_{fit_type}_{N_bunches}_dt_r.npy')

profile.track()

print('Simulating...\n')

t_FF = np.linspace(0, rfstation.t_rev[0], h//5)
t_coarse = np.linspace(0, rfstation.t_rev[0], h)

for i in range(N_t):
    SPS_tracker.track()
    profile.track()
    OTFB.track()

    if (i + 1) % dtrack == 0:
        print(i + 1)

    if i % dplot == 0:
        plt.figure()
        plt.title('I_BEAM_COARSE_FF')
        plt.plot(t_FF, OTFB.OTFB_1.I_BEAM_COARSE_FF[-h//5:].real, color='r')
        #plt.plot(t_FF, OTFB.OTFB_1.I_BEAM_COARSE_FF[-h//5:].imag, color='b')
        plt.plot(t_coarse, OTFB.OTFB_1.I_COARSE_BEAM[-h:].real, color='r', linestyle='--')
        #plt.plot(t_coarse, OTFB.OTFB_1.I_COARSE_BEAM[-h:].imag, color='b', linestyle='--')

        plt.figure()
        plt.title('I_FF_CORR')
        plt.plot(OTFB.OTFB_1.I_FF_CORR.real, color='r')
        plt.plot(OTFB.OTFB_1.I_FF_CORR.imag, color='b')

        plt.figure()
        plt.title('V_FF_CORR')
        plt.plot(OTFB.OTFB_1.V_FF_CORR.real, color='r')
        plt.plot(OTFB.OTFB_1.V_FF_CORR.imag, color='b')

        plt.show()

    if i%dprof == 0:
        plt.figure()
        plt.title('Profile')
        plt.plot(profile.bin_centers, profile.n_macroparticles)
        plt.xlim((4.98e-6, 5.25e-6))
        plt.show()