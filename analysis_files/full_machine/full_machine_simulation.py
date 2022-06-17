'''
Simulation of a full SPS machine to compare with theoretical values.

Author: Birk Emil Karlsen-Bæck
'''

# Imports -------------------------------------------------------------------------------------------------------------
print('Importing...\n')
import numpy as np
import matplotlib.pyplot as plt

from full_machine_theoretical_estimates import theoretical_power, theoretical_signals

from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions import bigaussian
from blond.llrf.cavity_feedback import SPSCavityFeedback, CavityFeedbackCommissioning
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF

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
p_s = 450e9                                     # Synchronous momentum [eV]
h = 4620                                        # 200 MHz harmonic number [-]
V = 10e6                                        # 200 MHz RF voltage [V]
phi = 0                                         # 200 MHz phase [-]

# Parameters for the Simulation
N_t = 200#1000                                      # Number of turns to track

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
I_beam = np.max(np.real(OTFB.OTFB_1.I_COARSE_BEAM)/5)
P_wi_sim = [np.max(OTFB.OTFB_1.P_GEN), np.max(OTFB.OTFB_2.P_GEN)]
Vg_wi_sim = [np.max(OTFB.OTFB_1.V_IND_COARSE_GEN), np.max(OTFB.OTFB_2.V_IND_COARSE_GEN)]
Vb_wi_sim = [np.max(OTFB.OTFB_1.V_IND_COARSE_BEAM), np.max(OTFB.OTFB_2.V_IND_COARSE_BEAM)]
Va_wi_sim = [np.max(OTFB.OTFB_1.V_ANT), np.max(OTFB.OTFB_2.V_ANT)]
Ig_wi_sim = [np.max(OTFB.OTFB_1.I_GEN), np.max(OTFB.OTFB_2.I_GEN)]
MEAN = True
if MEAN:
    I_beam = np.max(np.real(OTFB.OTFB_1.I_COARSE_BEAM)/5)
    P_wi_sim = [np.mean(OTFB.OTFB_1.P_GEN), np.mean(OTFB.OTFB_2.P_GEN)]
    Vg_wi_sim = [np.mean(OTFB.OTFB_1.V_IND_COARSE_GEN), np.mean(OTFB.OTFB_2.V_IND_COARSE_GEN)]
    Vb_wi_sim = [np.mean(OTFB.OTFB_1.V_IND_COARSE_BEAM), np.mean(OTFB.OTFB_2.V_IND_COARSE_BEAM)]
    Va_wi_sim = [np.mean(OTFB.OTFB_1.V_ANT), np.mean(OTFB.OTFB_2.V_ANT)]
    Ig_wi_sim = [np.mean(OTFB.OTFB_1.I_GEN), np.mean(OTFB.OTFB_2.I_GEN)]



# Theoretical estimate:
f_r = [OTFB.OTFB_1.omega_r/(2 * np.pi), OTFB.OTFB_2.omega_r/(2 * np.pi)]
f_c = OTFB.OTFB_1.omega_c / (2 * np.pi)
R_beam = [OTFB.OTFB_1.TWC.R_beam, OTFB.OTFB_2.TWC.R_beam]
R_gen = [OTFB.OTFB_1.TWC.R_gen, OTFB.OTFB_2.TWC.R_gen]
tau = [OTFB.OTFB_1.TWC.tau, OTFB.OTFB_2.TWC.tau]
Vant = [V * V_part, V * (1 - V_part)]
n_cav = [4, 2]

print('f_r', f_r)
print('f_c', f_c)
print('R_beam',R_beam)
print('R_gen', R_gen)
print('tau', tau)
print(Vant)
print(n_cav)

P_wo, P_wi, Vg_wo, Vg_wi, Ig_wo, Ig_wi, Vb_wi = theoretical_power(f_r, f_c, R_beam, R_gen,
                                                                  tau, I_beam, Vant, n_cav,
                                                                  VOLT=True)

P_wo_ca, P_wi_ca, Vg_wo_ca, Vg_wi_ca, Ig_wo_ca, Ig_wi_ca, Vb_wi_ca = theoretical_power(f_r, f_c, R_beam, R_gen,
                                                                  tau, I_beam, Va_wi_sim, n_cav,
                                                                  VOLT=True, complex_antenna=True)



I_gen_wo1, I_gen_wi1 = theoretical_signals(OTFB.OTFB_1, I_beam)
I_gen_wo2, I_gen_wi2 = theoretical_signals(OTFB.OTFB_2, I_beam)
print(I_gen_wo1, I_gen_wi1)
print(I_gen_wo2, I_gen_wi2)

def f_e(sim, the):
    return 100 * np.abs(sim - the) / the

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
print('V_gen  ' + 7 * ' ' + f'{Vg_wi[0].real/1e6:.2f} MV' + 3 * ' ' + f'{Vg_wi[0].imag/1e6:.2f} MV' +
                   4 * ' ' + f'{Vg_wi[1].real/1e6:.2f} MV' + 3 * ' ' + f'{Vg_wi[1].imag/1e6:.2f} MV')
print('I_gen wo' + 7 * ' ' + f'{Ig_wo[0].real:.2f} A' + 3 * ' ' + f'{Ig_wo[0].imag:.2f} A' +
                   4 * ' ' + f'{Ig_wo[1].real:.2f} A' + 3 * ' ' + f'{Ig_wo[1].imag:.2f} A')
print('I_gen wi' + 7 * ' ' + f'{Ig_wi[0].real:.2f} A' + 3 * ' ' + f'{Ig_wi[0].imag:.2f} A' +
                   4 * ' ' + f'{Ig_wi[1].real:.2f} A' + 3 * ' ' + f'{Ig_wi[1].imag:.2f} A')


print('\n\n------ Simulation Estimates ------')
print('Cavity' + 14 * ' ' + '3-section' + 14 * ' ' + '4-section')
print('Config' + 10 * ' ' + 'In-ph' + 5 * ' ' + 'Quadr' + 7 * ' ' + 'In-ph' + 5 * ' ' + 'Quadr')
print('V_beam  ' + 7 * ' ' + f'{Vb_wi_sim[0].real/1e6:.2f} MV' + 3 * ' ' + f'{Vb_wi_sim[0].imag/1e6:.2f} MV' +
                   4 * ' ' + f'{Vb_wi_sim[1].real/1e6:.2f} MV' + 3 * ' ' + f'{Vb_wi_sim[1].imag/1e6:.2f} MV')
print('V_gen  ' + 7 * ' ' + f'{Vg_wi_sim[0].real/1e6:.2f} MV' + 3 * ' ' + f'{Vg_wi_sim[0].imag/1e6:.2f} MV' +
                   4 * ' ' + f'{Vg_wi_sim[1].real/1e6:.2f} MV' + 3 * ' ' + f'{Vg_wi_sim[1].imag/1e6:.2f} MV')
print('I_gen wo' + 7 * ' ' + f'{Ig_wo_sim[0].real:.2f} A' + 3 * ' ' + f'{Ig_wo_sim[0].imag:.2f} A' +
                   4 * ' ' + f'{Ig_wo_sim[1].real:.2f} A' + 3 * ' ' + f'{Ig_wo_sim[1].imag:.2f} A')
print('I_gen wi' + 7 * ' ' + f'{Ig_wi_sim[0].real:.2f} A' + 3 * ' ' + f'{Ig_wi_sim[0].imag:.2f} A' +
                   4 * ' ' + f'{Ig_wi_sim[1].real:.2f} A' + 3 * ' ' + f'{Ig_wi_sim[1].imag:.2f} A')
print('I_gen wi', np.angle(Ig_wi_sim[0], deg=True), np.angle(Ig_wi_sim[1], deg=True))
print('V_gen wi', np.angle(Vg_wi_sim[0], deg=True), np.angle(Vg_wi_sim[1], deg=True))


print('\n\n------ Theory Estimates Complex Antenna ------')
print('Cavity' + 14 * ' ' + '3-section' + 14 * ' ' + '4-section')
print('Config' + 10 * ' ' + 'In-ph' + 5 * ' ' + 'Quadr' + 7 * ' ' + 'In-ph' + 5 * ' ' + 'Quadr')
print('V_beam  ' + 7 * ' ' + f'{Vb_wi_ca[0].real/1e6:.2f} MV' + 3 * ' ' + f'{Vb_wi_ca[0].imag/1e6:.2f} MV' +
                   4 * ' ' + f'{Vb_wi_ca[1].real/1e6:.2f} MV' + 3 * ' ' + f'{Vb_wi_ca[1].imag/1e6:.2f} MV')
print('V_gen  ' + 7 * ' ' + f'{Vg_wi_ca[0].real/1e6:.2f} MV' + 3 * ' ' + f'{Vg_wi_ca[0].imag/1e6:.2f} MV' +
                   4 * ' ' + f'{Vg_wi_ca[1].real/1e6:.2f} MV' + 3 * ' ' + f'{Vg_wi_ca[1].imag/1e6:.2f} MV')
print('I_gen wo' + 7 * ' ' + f'{Ig_wo_ca[0].real:.2f} A' + 3 * ' ' + f'{Ig_wo_ca[0].imag:.2f} A' +
                   4 * ' ' + f'{Ig_wo_ca[1].real:.2f} A' + 3 * ' ' + f'{Ig_wo_ca[1].imag:.2f} A')
print('I_gen wi' + 7 * ' ' + f'{Ig_wi_ca[0].real:.2f} A' + 3 * ' ' + f'{Ig_wi_ca[0].imag:.2f} A' +
                   4 * ' ' + f'{Ig_wi_ca[1].real:.2f} A' + 3 * ' ' + f'{Ig_wi_ca[1].imag:.2f} A')
print('I_gen wi', np.angle(Ig_wi_ca[0], deg=True), np.angle(Ig_wi_ca[1], deg=True))
print('V_gen wi', np.angle(Vg_wi_ca[0], deg=True), np.angle(Vg_wi_ca[1], deg=True))



print('\n\n------ Power Estimates Complex Antenna ------')
print('Cavity' + 14 * ' ' + '3-section' + 14 * ' ' + '4-section')
print('Config' + 10 * ' ' + 'Theory' + 5 * ' ' + 'Model' + 7 * ' ' + 'Theory' + 5 * ' ' + 'Model')
print('Power wo' + 7 * ' ' + f'{P_wo_ca[0]/1e3:.1f} kW' + 3 * ' ' + f'{P_wo_sim[0]/1e3:.1f} kW' +
                   4 * ' ' + f'{P_wo_ca[1]/1e3:.1f} kW' + 3 * ' ' + f'{P_wo_sim[1]/1e3:.1f} kW')
print('Power wi' + 7 * ' ' + f'{P_wi_ca[0]/1e3:.1f} kW' + 3 * ' ' + f'{P_wi_sim[0]/1e3:.1f} kW' +
                   3 * ' ' + f'{P_wi_ca[1]/1e3:.1f} kW' + 3 * ' ' + f'{P_wi_sim[1]/1e3:.1f} kW')


print()
print('#############################################')
print('########### Agreement with Theory ###########')
print('#############################################')


print('\n\n------ Power Estimates ------')
print('Cavity' + 14 * ' ' + '3-section' + 14 * ' ' + '4-section')
print('Config' + 10 * ' ' + 'Theory' + 5 * ' ' + 'Model' + 7 * ' ' + 'Theory' + 5 * ' ' + 'Model')
print('Power wo' + 7 * ' ' + f'{P_wo[0]/1e3:.1f} kW' + 3 * ' ' + f'{f_e(P_wo_sim[0], P_wo[0]) :.2f} %' +
                   4 * ' ' + f'{P_wo[1]/1e3:.1f} kW' + 3 * ' ' + f'{f_e(P_wo_sim[1], P_wo[1]) :.2f} %')
print('Power wi' + 7 * ' ' + f'{P_wi[0]/1e3:.1f} kW' + 3 * ' ' + f'{f_e(P_wi_sim[0], P_wi[0]) :.2f} %' +
                   3 * ' ' + f'{P_wi[1]/1e3:.1f} kW' + 3 * ' ' + f'{f_e(P_wi_sim[1], P_wi[1]) :.2f} %')


print('\n\n------ Theory Estimates ------')
print('Cavity' + 14 * ' ' + '3-section' + 14 * ' ' + '4-section')
print('Config' + 10 * ' ' + 'In-ph' + 5 * ' ' + 'Quadr' + 7 * ' ' + 'In-ph' + 5 * ' ' + 'Quadr')
print('V_beam  ' + 7 * ' ' + f'{f_e(Vb_wi_sim[0].real, Vb_wi[0].real):.2f} %'
                 + 3 * ' ' + f'{f_e(Vb_wi_sim[0].imag, Vb_wi[0].imag):.2f} %'
                 + 4 * ' ' + f'{f_e(Vb_wi_sim[1].real, Vb_wi[1].real):.2f} %'
                 + 3 * ' ' + f'{f_e(Vb_wi_sim[1].imag, Vb_wi[1].imag):.2f} %')
print('V_gen  '  + 7 * ' ' + f'{f_e(Vg_wi_sim[0].real, Vg_wi[0].real):.2f} %'
                 + 3 * ' ' + f'{f_e(Vg_wi_sim[0].imag, Vg_wi[0].imag):.2f} %'
                 + 4 * ' ' + f'{f_e(Vg_wi_sim[1].real, Vg_wi[1].real):.2f} %'
                 + 3 * ' ' + f'{f_e(Vg_wi_sim[1].imag, Vg_wi[1].imag):.2f} %')
print('I_gen wo' + 7 * ' ' + f'{f_e(Ig_wo_sim[0].real, Ig_wo[0].real):.2f} %'
                 + 3 * ' ' + f'{f_e(Ig_wo_sim[0].imag, Ig_wo[0].imag):.2f} %'
                 + 4 * ' ' + f'{f_e(Ig_wo_sim[1].real, Ig_wo[1].real):.2f} %'
                 + 3 * ' ' + f'{f_e(Ig_wo_sim[1].imag, Ig_wo[1].imag):.2f} %')
print('I_gen wi' + 7 * ' ' + f'{f_e(Ig_wi_sim[0].real, Ig_wi[0].real):.2f} %'
                 + 3 * ' ' + f'{f_e(Ig_wi_sim[0].imag, Ig_wi[0].imag):.2f} %'
                 + 4 * ' ' + f'{f_e(Ig_wi_sim[1].real, Ig_wi[1].real):.2f} %'
                 + 3 * ' ' + f'{f_e(Ig_wi_sim[1].imag, Ig_wi[1].imag):.2f} %')
print('I_gen wi', f_e(np.abs(Ig_wi_sim[0]), np.abs(Ig_wi[0])), f_e(np.abs(Ig_wi_sim[1]), np.abs(Ig_wi[1])))
print('V_gen wi', f_e(np.abs(Vg_wi_sim[0]), np.abs(Vg_wi[0])), f_e(np.abs(Vg_wi_sim[1]), np.abs(Vg_wi[1])))



print('\n\n------ Theory Estimates Complex Antenna ------')
print('Cavity' + 14 * ' ' + '3-section' + 14 * ' ' + '4-section')
print('Config' + 10 * ' ' + 'In-ph' + 5 * ' ' + 'Quadr' + 7 * ' ' + 'In-ph' + 5 * ' ' + 'Quadr')
print('V_beam  ' + 7 * ' ' + f'{f_e(Vb_wi_sim[0].real, Vb_wi_ca[0].real):.2f} %'
                 + 3 * ' ' + f'{f_e(Vb_wi_sim[0].imag, Vb_wi_ca[0].imag):.2f} %'
                 + 4 * ' ' + f'{f_e(Vb_wi_sim[1].real, Vb_wi_ca[1].real):.2f} %'
                 + 3 * ' ' + f'{f_e(Vb_wi_sim[1].imag, Vb_wi_ca[1].imag):.2f} %')
print('V_gen  '  + 7 * ' ' + f'{f_e(Vg_wi_sim[0].real, Vg_wi_ca[0].real):.2f} %'
                 + 3 * ' ' + f'{f_e(Vg_wi_sim[0].imag, Vg_wi_ca[0].imag):.2f} %'
                 + 4 * ' ' + f'{f_e(Vg_wi_sim[1].real, Vg_wi_ca[1].real):.2f} %'
                 + 3 * ' ' + f'{f_e(Vg_wi_sim[1].imag, Vg_wi_ca[1].imag):.2f} %')
print('I_gen wo' + 7 * ' ' + f'{f_e(Ig_wo_sim[0].real, Ig_wo_ca[0].real):.2f} %'
                 + 3 * ' ' + f'{f_e(Ig_wo_sim[0].imag, Ig_wo_ca[0].imag):.2f} %'
                 + 4 * ' ' + f'{f_e(Ig_wo_sim[1].real, Ig_wo_ca[1].real):.2f} %'
                 + 3 * ' ' + f'{f_e(Ig_wo_sim[1].imag, Ig_wo_ca[1].imag):.2f} %')
print('I_gen wi' + 7 * ' ' + f'{f_e(Ig_wi_sim[0].real, Ig_wi_ca[0].real):.2f} %'
                 + 3 * ' ' + f'{f_e(Ig_wi_sim[0].imag, Ig_wi_ca[0].imag):.2f} %'
                 + 4 * ' ' + f'{f_e(Ig_wi_sim[1].real, Ig_wi_ca[1].real):.2f} %'
                 + 3 * ' ' + f'{f_e(Ig_wi_sim[1].imag, Ig_wi_ca[1].imag):.2f} %')
print('I_gen wi', f_e(np.abs(Ig_wi_sim[0]), np.abs(Ig_wi_ca[0])), f_e(np.abs(Ig_wi_sim[1]), np.abs(Ig_wi_ca[1])))
print('V_gen wi', f_e(np.abs(Vg_wi_sim[0]), np.abs(Vg_wi_ca[0])), f_e(np.abs(Vg_wi_sim[1]), np.abs(Vg_wi_ca[1])))



print('\n\n------ Power Estimates Complex Antenna ------')
print('Cavity' + 14 * ' ' + '3-section' + 14 * ' ' + '4-section')
print('Config' + 10 * ' ' + 'Theory' + 5 * ' ' + 'Model' + 7 * ' ' + 'Theory' + 5 * ' ' + 'Model')
print('Power wo' + 7 * ' ' + f'{P_wo_ca[0]/1e3:.1f} kW' + 3 * ' ' + f'{f_e(P_wo_sim[0], P_wo_ca[0]) :.2f} %' +
                   4 * ' ' + f'{P_wo_ca[1]/1e3:.1f} kW' + 3 * ' ' + f'{f_e(P_wo_sim[1], P_wo_ca[1]) :.2f} %')
print('Power wi' + 7 * ' ' + f'{P_wi_ca[0]/1e3:.1f} kW' + 3 * ' ' + f'{f_e(P_wi_sim[0], P_wi_ca[0]) :.2f} %' +
                   3 * ' ' + f'{P_wi_ca[1]/1e3:.1f} kW' + 3 * ' ' + f'{f_e(P_wi_sim[1], P_wi_ca[1]) :.2f} %')



print(np.mean(OTFB.OTFB_1.I_COARSE_BEAM))

plt.figure()
plt.plot([0, Ig_wo_sim[0].real], [0, Ig_wo_sim[0].imag], label=r'$Ig$ wo sim')
plt.plot([0, Ig_wi_sim[0].real], [0, Ig_wi_sim[0].imag], label=r'$Ig$ wi sim')
plt.plot([0, Ig_wo[0].real], [0, Ig_wo[0].imag], label=r'$Ig$ wo')
plt.plot([0, Ig_wi[0].real], [0, Ig_wi[0].imag], label=r'$Ig$ wi')
plt.plot([0, Ig_wo_ca[0].real], [0, Ig_wo_ca[0].imag], label=r'$Ig$ wo ca')
plt.plot([0, Ig_wi_ca[0].real], [0, Ig_wi_ca[0].imag], label=r'$Ig$ wi ca')

plt.legend()

plt.figure()
plt.plot([0, Ig_wo_sim[1].real], [0, Ig_wo_sim[1].imag], label=r'$Ig$ wo sim')
plt.plot([0, Ig_wi_sim[1].real], [0, Ig_wi_sim[1].imag], label=r'$Ig$ wi sim')
plt.plot([0, Ig_wo[1].real], [0, Ig_wo[1].imag], label=r'$Ig$ wo')
plt.plot([0, Ig_wi[1].real], [0, Ig_wi[1].imag], label=r'$Ig$ wi')
plt.plot([0, Ig_wo_ca[1].real], [0, Ig_wo_ca[1].imag], label=r'$Ig$ wo ca')
plt.plot([0, Ig_wi_ca[1].real], [0, Ig_wi_ca[1].imag], label=r'$Ig$ wi ca')
plt.legend()






fig, ax = plt.subplots(2, 1, figsize=(6, 6))

V_s = 1e-6
Xf1 = 5.5
Xf2 = 5.5

ax[0].set_title('3-section')
# Simulation
ax[0].plot([Vb_wi_sim[0].real * V_s, 0], [Vb_wi_sim[0].imag * V_s, 0],
           label=r'Sim $V_b$', color='r', linestyle='--',
           markevery=2, marker='<', markersize=10)
ax[0].plot([Vg_wi_sim[0].real * V_s, 0], [Vg_wi_sim[0].imag * V_s, 0],
           label=r'Sim $V_g$', color='r', linestyle='-.',
           markevery=2, marker='<', markersize=10)
ax[0].plot([Va_wi_sim[0].real * V_s, 0], [Va_wi_sim[0].imag * V_s, 0],
           label=r'Sim $V_a$', color='r',
           markevery=2, marker='<', markersize=10)
ax[0].grid()
# Theory
ax[0].plot([Vb_wi[0].real * V_s, 0], [Vb_wi[0].imag * V_s, 0],
           label=r'Theory $V_b$', color='b', linestyle='--',
           markevery=2, marker='>', markersize=10)
ax[0].plot([Vg_wi[0].real * V_s, 0], [Vg_wi[0].imag * V_s, 0],
           label=r'Theory $V_g$', color='b', linestyle='-.',
           markevery=2, marker='>', markersize=10)
ax[0].plot([0, 0], [Vant[0] * V_s, 0],
           label=r'Theory $V_a$', color='b',
           markevery=2, marker='>', markersize=10)
ax[0].set_xlabel('In-phase [MV]')
ax[0].set_ylabel('Quadrature [MV]')
ax[0].set_xlim((-Xf1, Xf1))
ax[0].set_ylim((0, Xf1))


ax[1].set_title('4-section')
# Simulation
ax[1].plot([Vb_wi_sim[1].real * V_s, 0], [Vb_wi_sim[1].imag * V_s, 0],
           label=r'Sim $V_b$', color='r', linestyle='--',
           markevery=2, marker='<', markersize=10)
ax[1].plot([Vg_wi_sim[1].real * V_s, 0], [Vg_wi_sim[1].imag * V_s, 0],
           label=r'Sim $V_g$', color='r', linestyle='-.',
           markevery=2, marker='<', markersize=10)
ax[1].plot([Va_wi_sim[1].real * V_s, 0], [Va_wi_sim[1].imag * V_s, 0],
           label=r'Sim $V_a$', color='r',
           markevery=2, marker='<', markersize=10)
ax[1].grid()
# Theory
ax[1].plot([Vb_wi[1].real * V_s, 0], [Vb_wi[1].imag * V_s, 0],
           label=r'Theory $V_b$', color='b', linestyle='--',
           markevery=2, marker='>', markersize=10)
ax[1].plot([Vg_wi[1].real * V_s, 0], [Vg_wi[1].imag * V_s, 0],
           label=r'Theory $V_g$', color='b', linestyle='-.',
           markevery=2, marker='>', markersize=10)
ax[1].plot([0, 0], [Vant[1] * V_s, 0],
           label=r'Theory $V_a$', color='b',
           markevery=2, marker='>', markersize=10)
ax[1].set_xlabel('In-phase [MV]')
ax[1].set_ylabel('Quadrature [MV]')
ax[1].set_xlim((-Xf2, Xf2))
ax[1].set_ylim((0, Xf2))

handles, labels = ax[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.5), ncol=3)






fig, ax = plt.subplots(2, 1, figsize=(6, 6))

V_s = 1e-6
Xf1 = 5.5
Xf2 = 5.5

ax[0].set_title('3-section')
# Simulation
ax[0].plot([Vb_wi_sim[0].real * V_s, 0], [Vb_wi_sim[0].imag * V_s, 0],
           label=r'Sim $V_b$', color='r', linestyle='--',
           markevery=2, marker='<', markersize=10)
ax[0].plot([Vg_wi_sim[0].real * V_s, 0], [Vg_wi_sim[0].imag * V_s, 0],
           label=r'Sim $V_g$', color='r', linestyle='-.',
           markevery=2, marker='<', markersize=10)
ax[0].plot([Va_wi_sim[0].real * V_s, 0], [Va_wi_sim[0].imag * V_s, 0],
           label=r'Sim $V_a$', color='r',
           markevery=2, marker='<', markersize=10)
ax[0].grid()
# Theory
ax[0].plot([Vb_wi_ca[0].real * V_s, 0], [Vb_wi_ca[0].imag * V_s, 0],
           label=r'Theory $V_b$', color='b', linestyle='--',
           markevery=2, marker='>', markersize=10)
ax[0].plot([Vg_wi_ca[0].real * V_s, 0], [Vg_wi_ca[0].imag * V_s, 0],
           label=r'Theory $V_g$', color='b', linestyle='-.',
           markevery=2, marker='>', markersize=10)
ax[0].plot([0, 0], [Vant[0] * V_s, 0],
           label=r'Theory $V_a$', color='b',
           markevery=2, marker='>', markersize=10)
ax[0].set_xlabel('In-phase [MV]')
ax[0].set_ylabel('Quadrature [MV]')
ax[0].set_xlim((-Xf1, Xf1))
ax[0].set_ylim((0, Xf1))


ax[1].set_title('4-section')
# Simulation
ax[1].plot([Vb_wi_sim[1].real * V_s, 0], [Vb_wi_sim[1].imag * V_s, 0],
           label=r'Sim $V_b$', color='r', linestyle='--',
           markevery=2, marker='<', markersize=10)
ax[1].plot([Vg_wi_sim[1].real * V_s, 0], [Vg_wi_sim[1].imag * V_s, 0],
           label=r'Sim $V_g$', color='r', linestyle='-.',
           markevery=2, marker='<', markersize=10)
ax[1].plot([Va_wi_sim[1].real * V_s, 0], [Va_wi_sim[1].imag * V_s, 0],
           label=r'Sim $V_a$', color='r',
           markevery=2, marker='<', markersize=10)
ax[1].grid()
# Theory
ax[1].plot([Vb_wi_ca[1].real * V_s, 0], [Vb_wi_ca[1].imag * V_s, 0],
           label=r'Theory $V_b$', color='b', linestyle='--',
           markevery=2, marker='>', markersize=10)
ax[1].plot([Vg_wi_ca[1].real * V_s, 0], [Vg_wi_ca[1].imag * V_s, 0],
           label=r'Theory $V_g$', color='b', linestyle='-.',
           markevery=2, marker='>', markersize=10)
ax[1].plot([0, 0], [Vant[1] * V_s, 0],
           label=r'Theory $V_a$', color='b',
           markevery=2, marker='>', markersize=10)
ax[1].set_xlabel('In-phase [MV]')
ax[1].set_ylabel('Quadrature [MV]')
ax[1].set_xlim((-Xf2, Xf2))
ax[1].set_ylim((0, Xf2))

handles, labels = ax[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.5), ncol=3)


plt.show()