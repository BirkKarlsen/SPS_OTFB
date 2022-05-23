'''
Simulation of a full SPS machine to compare with theoretical values.

Author: Birk Emil Karlsen-Bæck
'''

# Imports -------------------------------------------------------------------------------------------------------------
print('Importing...\n')
import numpy as np
import matplotlib.pyplot as plt

from full_machine_theoretical_estimates import theoretical_power

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
p_s = 450e9                                     # Synchronous momentum [eV]
h = 4620                                        # 200 MHz harmonic number [-]
V = 10e6                                        # 200 MHz RF voltage [V]
phi = 0                                         # 200 MHz phase [-]

# Parameters for the Simulation
N_t = 1000                                      # Number of turns to track

# SPS Cavity Controler Parameters
df = [0.18433333e6,
      0.2275e6]
G_tx = [0.3/0.33,
        0.3/0.33]
G_llrf = 20
a_comb = 63/64
V_part = 0.5172

# Beam parameters
N_m = int(5e5)                                  # Number of macro-particles for tracking
N_p = 2.3e11                                    # Number of protons per bunch
spacing = 5                                     # Spacing between bunches in units of buckets
N_bunches = h // 5                              # Number of bunches
bl = 1.2e-9                                     # Bunch length [s]


# Objects -------------------------------------------------------------------------------------------------------------
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
Commissioning = CavityFeedbackCommissioning(open_FF=True, debug=False, rot_IQ=-1)
OTFB = SPSCavityFeedback(rfstation, beam, profile, post_LS2=True, Commissioning=Commissioning,
                         G_tx=G_tx, a_comb=a_comb, G_llrf=G_llrf, df=df, V_part=V_part)


# Tracker Object without SPS OTFB
SPS_rf_tracker = RingAndRFTracker(rfstation, beam, CavityFeedback=OTFB,
                                  Profile=profile, interpolation=True)
SPS_tracker = FullRingAndRF([SPS_rf_tracker])

profile.track()


# Tracking ------------------------------------------------------------------------------------------------------------
print('Simulating...\n')
dt_track = 50

# Values without beam
OTFB.OTFB_1.calc_power()
OTFB.OTFB_2.calc_power()
P_wo_sim = [np.max(OTFB.OTFB_1.P_GEN), np.max(OTFB.OTFB_2.P_GEN)]
Vg_wo_sim = [np.max(OTFB.OTFB_1.V_IND_COARSE_GEN), np.max(OTFB.OTFB_2.V_IND_COARSE_GEN)]
Ig_wo_sim = [np.max(OTFB.OTFB_1.I_GEN), np.max(OTFB.OTFB_2.I_GEN)]

for i in range(N_t):
    # Assuming a static beam and therefore only tracking the OTFB
    OTFB.track()

    if i % dt_track == 0:
        print('turn:', i)
        print('Peak RF beam current: ' + f'{np.max(np.abs(OTFB.OTFB_1.I_COARSE_BEAM))/5:.2f} A')
        print('Peak beam-induced voltage: ' + f'{np.max(np.abs(OTFB.OTFB_1.V_IND_COARSE_BEAM))/1e6:.3f} MV')
        print('Peak generator-induced voltage: ' + f'{np.max(np.abs(OTFB.OTFB_1.V_IND_COARSE_GEN))/1e6:.3f} MV')


OTFB.OTFB_1.calc_power()
OTFB.OTFB_2.calc_power()
I_beam = np.max(np.abs(OTFB.OTFB_1.I_COARSE_BEAM)/5)
P_wi_sim = [np.max(OTFB.OTFB_1.P_GEN), np.max(OTFB.OTFB_2.P_GEN)]
Vg_wi_sim = [np.max(OTFB.OTFB_1.V_IND_COARSE_GEN), np.max(OTFB.OTFB_1.V_IND_COARSE_GEN)]
Vb_wi_sim = [np.max(OTFB.OTFB_1.V_IND_COARSE_BEAM), np.max(OTFB.OTFB_2.V_IND_COARSE_BEAM)]
Ig_wi_sim = [np.max(OTFB.OTFB_1.I_GEN), np.max(OTFB.OTFB_2.I_GEN)]


# Theoretical estimate:
f_r = [OTFB.OTFB_1.omega_r/(2 * np.pi), OTFB.OTFB_2.omega_r/(2 * np.pi)]
f_c = OTFB.OTFB_1.omega_c / (2 * np.pi)
R_beam = [OTFB.OTFB_1.TWC.R_beam, OTFB.OTFB_2.TWC.R_beam]
R_gen = [OTFB.OTFB_1.TWC.R_gen, OTFB.OTFB_2.TWC.R_gen]
tau = [OTFB.OTFB_1.TWC.tau, OTFB.OTFB_2.TWC.tau]
Vant = [V * V_part, V * (1 - V_part)]
n_cav = [4, 2]

P_wo, P_wi, Vg_wo, Vg_wi, Ig_wo, Ig_wi, Vb_wi = theoretical_power(f_r, f_c, R_beam, R_gen, tau, I_beam, Vant, n_cav, VOLT=True)

# Print Values
print('\n\n------ Power Estimates ------')
print('Cavity' + 14 * ' ' + '3-section' + 14 * ' ' + '4-section')
print('Config' + 10 * ' ' + 'Theory' + 5 * ' ' + 'Model' + 7 * ' ' + 'Theory' + 5 * ' ' + 'Model')
print('Power wo' + 7 * ' ' + f'{P_wo[0]/1e3:.1f} kW' + 3 * ' ' + f'{P_wo_sim[0]/1e3:.1f} kW' +
                   4 * ' ' + f'{P_wo[1]/1e3:.1f} kW' + 3 * ' ' + f'{P_wo_sim[1]/1e3:.1f} kW')
print('Power wi' + 7 * ' ' + f'{P_wi[0]/1e3:.1f} kW' + 3 * ' ' + f'{P_wi_sim[0]/1e3:.1f} kW' +
                   3 * ' ' + f'{P_wi[1]/1e3:.1f} kW' + 3 * ' ' + f'{P_wi_sim[1]/1e3:.1f} kW')

print('\n\n------ Theory Estimates ------')
print('Cavity' + 14 * ' ' + '3-section' + 14 * ' ' + '4-section')
print('Config' + 10 * ' ' + 'In-ph' + 5 * ' ' + 'Quadr' + 7 * ' ' + 'In-ph' + 5 * ' ' + 'Quadr')
print('V_beam  ' + 7 * ' ' + f'{Vb_wi[0].real/1e6:.2f} MV' + 3 * ' ' + f'{Vb_wi[0].imag/1e6:.2f} MV' +
                   4 * ' ' + f'{Vb_wi[1].real/1e6:.2f} MV' + 3 * ' ' + f'{Vb_wi[1].imag/1e6:.2f} MV')
print('I_gen wo' + 7 * ' ' + f'{Ig_wo[0].real:.2f} A' + 3 * ' ' + f'{Ig_wo[0].imag:.2f} A' +
                   4 * ' ' + f'{Ig_wo[1].real:.2f} A' + 3 * ' ' + f'{Ig_wo[1].imag:.2f} A')
print('I_gen wi' + 7 * ' ' + f'{Ig_wi[0].real:.2f} A' + 3 * ' ' + f'{Ig_wi[0].imag:.2f} A' +
                   4 * ' ' + f'{Ig_wi[1].real:.2f} A' + 3 * ' ' + f'{Ig_wi[1].imag:.2f} A')


print('\n\n------ Simulation Estimates ------')
print('Cavity' + 14 * ' ' + '3-section' + 14 * ' ' + '4-section')
print('Config' + 10 * ' ' + 'In-ph' + 5 * ' ' + 'Quadr' + 7 * ' ' + 'In-ph' + 5 * ' ' + 'Quadr')
print('V_beam  ' + 7 * ' ' + f'{Vb_wi_sim[0].real/1e6:.2f} MV' + 3 * ' ' + f'{Vb_wi_sim[0].imag/1e6:.2f} MV' +
                   4 * ' ' + f'{Vb_wi_sim[1].real/1e6:.2f} MV' + 3 * ' ' + f'{Vb_wi_sim[1].imag/1e6:.2f} MV')
print('I_gen wo' + 7 * ' ' + f'{Ig_wo_sim[0].real:.2f} A' + 3 * ' ' + f'{Ig_wo_sim[0].imag:.2f} A' +
                   4 * ' ' + f'{Ig_wo_sim[1].real:.2f} A' + 3 * ' ' + f'{Ig_wo_sim[1].imag:.2f} A')
print('I_gen wi' + 7 * ' ' + f'{Ig_wi_sim[0].real:.2f} A' + 3 * ' ' + f'{Ig_wi_sim[0].imag:.2f} A' +
                   4 * ' ' + f'{Ig_wi_sim[1].real:.2f} A' + 3 * ' ' + f'{Ig_wi_sim[1].imag:.2f} A')
