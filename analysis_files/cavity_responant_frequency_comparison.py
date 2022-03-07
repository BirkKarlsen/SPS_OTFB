'''
This file was made to compare the behaviour of the OTFB based on the responant/central frequency of the cavity.

Author: Birk Emil Karlsen-Bæck
'''

# Imports ---------------------------------------------------------------------
print('Importing...\n')
import numpy as np
import matplotlib.pyplot as plt


from blond.llrf.cavity_feedback import SPSCavityFeedback, CavityFeedbackCommissioning
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions_multibunch import matched_from_distribution_density_multibunch
from blond.trackers.tracker import FullRingAndRF, RingAndRFTracker
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from blond.impedances.impedance_sources import InputTable

# Parameters ------------------------------------------------------------------
C = 2 * np.pi * 1100.009                        # Ring circumference [m]
gamma_t = 18.0                                  # Transition Gamma [-]
alpha = 1 / (gamma_t**2)                        # Momentum compaction factor [-]
p_s = 440e9                                     # Synchronous momentum [eV]
h = 4620                                        # 200 MHz harmonic number [-]
V = (0.911535 * 4 + 1.526871 * 2) * 1e6         # 200 MHz RF voltage [V]
phi = 0                                         # 200 MHz phase [-]

# Parameters for the SPS Cavity Feedback
G_llrf = 16
V_part = 0.5442095845867135                     # Voltage partitioning [-]
# For OTFB at measured frequencies
rr = 0.7
G_tx_1 = [0.163212561182363 * rr,
          0.127838041632473 * rr]
df_1 = [0,
        0]

# For OTFB at 200.222 MHz
G_tx_2 = [0.229377820916177,
          0.430534529571209]
df_2 = [0.18433333e6,
        0.2275e6]



# Parameters for the Simulation
N_m = int(5e5)                                  # Number of macro-particles for tracking
N_t = 1000                                      # Number of turns to track

total_intensity = 3385.8196 * 10**10
lxdir = '../'
N_bunches = 288
fit_type= 'fwhm'

# Objects ---------------------------------------------------------------------
print('Setting up...\n')

# SPS Ring
ring_1 = Ring(C, alpha, p_s, Proton(), n_turns=N_t)
ring_2 = Ring(C, alpha, p_s, Proton(), n_turns=N_t)


# RF Station
rfstation_1 = RFStation(ring_1, [h, 4 * h], [V, 0.19 * V], [0, np.pi], n_rf=2)
rfstation_2 = RFStation(ring_2, [h, 4 * h], [V, 0.19 * V], [0, np.pi], n_rf=2)
print(rfstation_1.omega_rf[0,0] / (2 * np.pi))


# Beam
bunch_intensities = np.load(lxdir + 'data_files/beam_parameters/avg_bunch_intensities_red.npy')
bunch_intensities = total_intensity * bunch_intensities / np.sum(bunch_intensities)  # normalize to 3385.8196 * 10**10
n_macro = N_m * N_bunches * bunch_intensities / np.sum(bunch_intensities)
beam_1 = Beam(ring_1, int(np.sum(n_macro[:N_bunches])), int(total_intensity))
beam_2 = Beam(ring_2, int(np.sum(n_macro[:N_bunches])), int(total_intensity))


# Profile
profile_1 = Profile(beam_1, CutOptions = CutOptions(cut_left=rfstation_1.t_rf[0,0] * (1000 - 2.5),
    cut_right=rfstation_1.t_rf[0,0] * (1000 + 72 * 5 * 4 + 250 * 3 + 125),
    n_slices=int(round(2**7 * (2.5 + 72 * 5 * 4 + 250 * 3 + 125)))))
profile_2 = Profile(beam_2, CutOptions = CutOptions(cut_left=rfstation_2.t_rf[0,0] * (1000 - 2.5),
    cut_right=rfstation_2.t_rf[0,0] * (1000 + 72 * 5 * 4 + 250 * 3 + 125),
    n_slices=int(round(2**7 * (2.5 + 72 * 5 * 4 + 250 * 3 + 125)))))


# SPS Cavity Feedback
Commissioning = CavityFeedbackCommissioning(open_FF=True, debug=False, rot_IQ=1)
OTFB_1 = SPSCavityFeedback(rfstation_1, beam_1, profile_1, post_LS2=True, V_part=V_part,
                         Commissioning=Commissioning, G_tx=G_tx_1, a_comb=31/32,
                         G_llrf=G_llrf, df=df_1)
OTFB_2 = SPSCavityFeedback(rfstation_2, beam_2, profile_2, post_LS2=True, V_part=V_part,
                         Commissioning=Commissioning, G_tx=G_tx_2, a_comb=31/32,
                         G_llrf=G_llrf, df=df_2)


# Tracker Object without SPS OTFB
SPS_rf_tracker_1 = RingAndRFTracker(rfstation_1, beam_1, CavityFeedback=OTFB_1,
                                    Profile=profile_1, interpolation=True)
SPS_tracker_1 = FullRingAndRF([SPS_rf_tracker_1])

SPS_rf_tracker_2 = RingAndRFTracker(rfstation_2, beam_2, CavityFeedback=OTFB_2,
                                    Profile=profile_2, interpolation=True)
SPS_tracker_2 = FullRingAndRF([SPS_rf_tracker_2])


beam_1.dE = np.load(
        lxdir + f'data_files/with_impedance/generated_beams/generated_beam_{fit_type}_{N_bunches}_dE_r.npy')
beam_1.dt = np.load(
        lxdir + f'data_files/with_impedance/generated_beams/generated_beam_{fit_type}_{N_bunches}_dt_r.npy')
beam_2.dE = np.load(
        lxdir + f'data_files/with_impedance/generated_beams/generated_beam_{fit_type}_{N_bunches}_dE_r.npy')
beam_2.dt = np.load(
        lxdir + f'data_files/with_impedance/generated_beams/generated_beam_{fit_type}_{N_bunches}_dt_r.npy')

profile_1.track()
profile_2.track()

SPS_tracker_1.track()
profile_1.track()
SPS_tracker_2.track()
profile_2.track()



print('Computing OTFB signals...\n')
n_t = 100
P_max_1 = np.zeros((n_t))
P_max_2 = np.zeros((n_t))
P_min_1 = np.zeros((n_t))
P_min_2 = np.zeros((n_t))
dtrack = 10
for i in range(n_t):
    OTFB_1.track()
    OTFB_2.track()

    OTFB_1.OTFB_1.calc_power()
    OTFB_1.OTFB_2.calc_power()
    OTFB_2.OTFB_1.calc_power()
    OTFB_2.OTFB_2.calc_power()

    P_max_1[i] = np.max(OTFB_1.OTFB_1.P_GEN[-h:])
    P_max_2[i] = np.max(OTFB_2.OTFB_1.P_GEN[-h:])

    P_min_1[i] = np.min(OTFB_1.OTFB_1.P_GEN[-h:])
    P_min_2[i] = np.min(OTFB_2.OTFB_1.P_GEN[-h:])

    if (i + 1)%dtrack == 0:
        print(i + 1)


print('Plotting...\n')

plt.figure()
plt.title('V_ANT')
plt.plot(np.abs(OTFB_1.OTFB_1.V_ANT[-h:]), label='meas')
plt.plot(np.abs(OTFB_2.OTFB_1.V_ANT[-h:]), label='200.222')
plt.legend()

plt.figure()
plt.title('V_BEAM')
plt.plot(np.abs(OTFB_1.OTFB_1.V_IND_COARSE_BEAM[-h:]), label='meas')
plt.plot(np.abs(OTFB_2.OTFB_1.V_IND_COARSE_BEAM[-h:]), label='200.222')
plt.legend()

plt.figure()
plt.title('V_GEN')
plt.plot(np.abs(OTFB_1.OTFB_1.V_IND_COARSE_GEN[-h:]), label='meas')
plt.plot(np.abs(OTFB_2.OTFB_1.V_IND_COARSE_GEN[-h:]), label='200.222')
plt.legend()

plt.figure()
plt.title('P_GEN')
plt.plot(OTFB_1.OTFB_1.P_GEN[-h:], label='meas')
plt.plot(OTFB_2.OTFB_1.P_GEN[-h:], label='200.222')
plt.legend()

plt.figure()
plt.plot(P_max_1, color='r')
plt.plot(P_min_1, color='r')
plt.plot(P_max_2, color='b')
plt.plot(P_min_2, color='b')

plt.show()