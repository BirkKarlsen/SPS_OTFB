'''
Python-script to optimize the transmitter-gains in the SPS

author: Birk Emil Karlsen-Bæck
'''
import argparse

parser = argparse.ArgumentParser(description="This file simulates the SPS OTFB with impedances.")

parser.add_argument("--bisect", '-bs', type=int,
                    help="Option to use the bisection script on this file, default is False (0)")
parser.add_argument("--g_tx_1", '-g1', type=float,
                    help="Option to input the transmitter gain for the 3-section")
parser.add_argument("--g_tx_2", '-g2', type=float,
                    help="Option to input the transmitter gain for the 4-section")


args = parser.parse_args()

# Options for script ----------------------------------------------------------
BISECT = False
SHOW_PLT = False
n_pretrack = 1000


# Imports ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import utility_files.analysis_tools as at

from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions import bigaussian
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.llrf.cavity_feedback import SPSCavityFeedback, CavityFeedbackCommissioning


# Parameters ------------------------------------------------------------------
C = 2 * np.pi * 1100.009                        # Ring circumference [m]
gamma_t = 18.0                                  # Transition Gamma [-]
alpha = 1 / (gamma_t**2)                        # Momentum compaction factor [-]
p_s = 440e9                                     # Synchronous momentum [eV]
h = 4620                                        # 200 MHz harmonic number [-]
#V = (0.911535 * 4 + 1.526871 * 2) * 1e6         # 200 MHz RF voltage [V]
V = 6660589.53641675
phi = 0                                         # 200 MHz phase [-]

N_m = int(5e5)                                  # Number of macro-particles for tracking
N_p = 1.15e11                                   # Number of particles
N_t = 1                                         # Number of turns to track
sigma_dt = 1.2e-9


# For a_comb of 63/64 and llrf gain of 20
a_comb = 31/32
G_llrf = 16
rr = 1
#V_part = 0.5442095845867135
V_part = 0.5517843967841601
#df = [0.18433333e6,        # Both at 200.222
#      0.2275e6]
#df = [62333.333,           # Both at 200.1
#      105500]
#df = [0,                   # Measured
#      0]
df = [0.71266666e6,         # Other side of omega_rf
      0.799e6]
G_tx = [0.163607060338826,
        0.1288941276502113]

if args.bisect is not None:
    BISECT = bool(args.bisect)
if args.g_tx_1 is not None:
    G_tx[0] = args.g_tx_1
if args.g_tx_2 is not None:
    G_tx[1] = args.g_tx_2


#df = [0, 0.2275e6]
#G_tx = [0.2607509145194842, 0.510893981556323]
#G_tx = [0.251402590786449, 0.511242728131293] For the old signs from before 02/02/2022 both at 200.222 MHz
#G_tx = [0.25154340605790062590, 0.510893981556323] # For the inverted real axis with 200.222 MHz
'''
For the PostLS2 scenario with both cavities at 200.222 MHz the optimized transmitter gains are
[0.251402590786449, 0.511242728131293]
'''

'''
Configurations and their optimized transmitter gains:

For a PostLS2 configuration with a_comb = 63/64, G_llrf = 20 and both cavity types at 200.222 MHz
we have
df = [0.18433333e6, 
      0.2275e6]
G_tx = [0.22909261332041,
        0.429420301179296]
        
For a PostLS2 configuration with a_comb = 31/32, G_llrf = 20 and both cavity types have the measured central frequency
we have
df = [0,
      0]
G_tx = [0.1611031942822209,
        0.115855991237277]
        
For a PostLS2 configuration with a_comb = 31/32, G_llrf = 16 and both cavity types have the measured central frequency
we have 
df = [0,
      0]
G_tx = [0.163212561182363,
        0.127838041632473]
For a PostLS2 configuration with a_comb = 31/32, G_llrf = 16 and both cavity types at 200.222 MHz
we have
df = [0.18433333e6,
      0.2275e6]
G_tx = [0.229377820916177,
        0.430534529571209]
For a PostLS2 configuration with a_comb = 63/64, Gllrf = 20 and both cavity types at measured frequencies
we have
df = [0,
      0]
G_tx = [0.1615069965527125,
        0.11584062194618076]
For a PostLS2 configuration with a_comb = 31/32, Gllrf = 16 and both cavity types at 200.1 MHz
we have
df = [62333.333,
      105500]
G_tx = [0.1910842957076554,
        0.289228143612504]
For a PostLS2 configuration with a_comb = 31/32, Gllrf = 0 and both cavity types at measured
we have
df = [0,
      0]
G_tx = [0.26041876200342555,
        0.5558826390544476]
For a PostLS2 configuration with a_comb = 31/32, Gllrf = 0 and both cavities at 200.222 MHz
we have
df = [0.18433333e6,
      0.2275e6]
G_tx = [0.25147316248903445,
        0.5110686163372516]
For a PostLS2 configuration with a_comb = 31/32, Gllrf = 0 and resonant frequencies at other side of omega_rf
we have
df = [0.71266666e6, 
      0.799e6]
G_tx = [0.2603649863930353,
        0.55564846411416855]
For a PostLS2 configuration with a_comb = 31/32, Gllrf = 16 and resonant frequencies at other side of omega_rf
we have
df = [0.71266666e6, 
      0.799e6]
G_tx = [0.163607060338826,
        0.1288941276502113]


'''

# Objects ---------------------------------------------------------------------

# SPS ring
ring = Ring(C, alpha, p_s, Proton(), N_t)

# RF-station
rfstation = RFStation(ring, [h], [V], [phi])

# Beam
beam = Beam(ring, N_m, N_p)

# Profile
profile = Profile(beam, CutOptions = CutOptions(cut_left=rfstation.t_rf[0,0]*(1000 + 0.1),
    cut_right=rfstation.t_rev[0], n_slices=2**7 * 4620))

bigaussian(ring, rfstation, beam, sigma_dt)


# SPS cavity controller
Commissioning = CavityFeedbackCommissioning(open_FF=True, debug=False)

OTFB = SPSCavityFeedback(rfstation, beam, profile, post_LS2=True,
                         Commissioning=Commissioning, G_tx=G_tx, a_comb=a_comb,
                         G_llrf=G_llrf, df=df, turns=n_pretrack, V_part=V_part)

if not BISECT:
    print(np.mean(np.angle(OTFB.OTFB_1.V_ANT)) * 180/np.pi)

    print(OTFB.OTFB_1.V_part, OTFB.OTFB_2.V_part)

# Comparison

target3 = V * 1e-6 * V_part
target4 = V * 1e-6 * (1 - V_part)

ant3 = np.mean(np.abs(OTFB.OTFB_1.V_ANT[-h:])) / 1e6
ant4 = np.mean(np.abs(OTFB.OTFB_2.V_ANT[-h:])) / 1e6

if not BISECT:
    print(f'Desired 3-section: {target3} MV')
    print(f'Model 3-section:   {ant3} MV')
    if target3 > ant3:
        print(f'Increase')
    else:
        print(f'Decrease')
    print()
    print(f'Desired 4-section: {target4} MV')
    print(f'Model 4-section:   {ant4} MV')
    if target4 > ant4:
        print(f'Increase')
    else:
        print(f'Decrease')

diff3 = np.abs(target3 - ant3) * 100 / target3
diff4 = np.abs(target4 - ant4) * 100 / target4

if not BISECT:
    print()
    print(f'3-section difference: {diff3} %')
    print(f'4-section difference: {diff4} %')

at.plot_OTFB_signals(OTFB.OTFB_1, h, rfstation.t_rf[0,0])
at.plot_OTFB_signals(OTFB.OTFB_2, h, rfstation.t_rf[0,0])

OTFB.OTFB_1.calc_power()
OTFB.OTFB_2.calc_power()

plt.figure()
plt.plot(OTFB.OTFB_1.P_GEN[-h:])
plt.plot(OTFB.OTFB_2.P_GEN[-h:])

if not BISECT:
    print()
    print('3-section power:', np.mean(OTFB.OTFB_1.P_GEN[-h:]))
    print('4-section power:', np.mean(OTFB.OTFB_2.P_GEN[-h:]))

if SHOW_PLT and not BISECT:
    plt.show()