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
SHOW_PLT = True
n_pretrack = 10000


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
p_s = 450e9                                     # Synchronous momentum [eV]
h = 4620                                        # 200 MHz harmonic number [-]
#V = (0.911535 * 4 + 1.526871 * 2) * 1e6         # 200 MHz RF voltage [V]
#V = 6660589.53641675
V = 10e6
phi = 0                                         # 200 MHz phase [-]

N_m = int(5e5)                                  # Number of macro-particles for tracking
N_p = 1.15e11                                   # Number of particles
N_t = 1                                         # Number of turns to track
sigma_dt = 1.2e-9


# For a_comb of 63/64 and llrf gain of 20
a_comb = 31/32
G_llrf = 20
rr = 1
#V_part = 0.5442095845867135
#V_part = 0.5517843967841601
V_part = 0.6
#df = [0.18433333e6,        # Both at 200.222
#      0.2275e6]
#df = [62333.333,           # Both at 200.1
#      105500]
df = [0,                   # Measured
      0]
#df = [0.71266666e6,         # Other side of omega_rf
#      0.799e6]
tr = 0.65
G_tx = [1 * 0.3 / 0.33,
        1 * 0.3 / 0.33]

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

-----------------------------
----- Newest OTFB model -----
-----------------------------

For PostLS2 configuration with a_comb = 63/64, Gllrf = 20 and TWCs at 200.222 MHz
df = [0.18433333e6,
      0.2275e6]
G_tx = [1.0355739238973907,
        1.078403005653143]
        
For PostLS2 configuration with a_comb = 31/32, Gllrf = 5 and TWCs at 200.222 MHz
df = [0.18433333e6,
      0.2275e6]
G_tx = [1.0317735694097596,
        1.0710520614580732]
        
For PostLS2 configuration with a_comb = 31/32, Gllrf = 10 and TWCs at 200.222 MHz
df = [0.18433333e6,
      0.2275e6]
G_tx = [1.0341903556357148,
        1.0757240557694563]
        
For PostLS2 configuration with a_comb = 31/32, Gllrf = 14 and TWCs at 200.222 MHz
df = [0.18433333e6,
      0.2275e6]
G_tx = [1.0349649477141394,
        1.0772235174474414]
        
For PostLS2 configuration with a_comb = 31/32, Gllrf = 16 and TWCs at 200.222 MHz
df = [0.18433333e6,
      0.2275e6]
G_tx = [1.0352156647332156,
        1.077709051028262]
        
For PostLS2 configuration with a_comb = 31/32, Gllrf = 20 and TWCs at 200.222 MHz
df = [0.18433333e6,
      0.2275e6]
G_tx = [1.035573923897336,
        1.0784030056522707]
        
For PostLS2 configuration with a_comb = 31/32 Gllrf = 5 and TWCs at 200.1 MHz
df = [62333.333,
      105500]
G_tx = [1.1067988416724432,
        1.2198386723730976]
        
For PostLS2 configuration with a_comb = 31/32 Gllrf = 10 and TWCs at 200.1 MHz
df = [62333.333,
      105500]
G_tx = [1.1146994926588105,
        1.2367097833279674]
        
For PostLS2 configuration with a_comb = 31/32 Gllrf = 14 and TWCs at 200.1 MHz
df = [62333.333,
      105500]
G_tx = [1.1172401859286643,
        1.2421617290781886]
        
For PostLS2 configuration witha_comb = 31/32, Gllrf = 16 and TWCs at 200.1 MHz
df = [62333.333,
      105500]
G_tx = [1.1180633496145078,
        1.2439306616383181]

For PostLS2 configuration with a_comb = 31/32, Gllrf = 20 and TWCs at 200.1 MHz
df = [62333.333,
      105500]
G_tx = [1.1192402524102003,
        1.246461850104049]

For PostLS2 configuration with a_comb = 31/32 Gllrf = 5 and TWCs at measured
df = [0,
      0]
G_tx = [1.1648584800986386,
        1.4564159619446317]

For PostLS2 configuration with a_comb = 31/32 Gllrf = 10 and TWCs at measured
df = [0,
      0]
G_tx = [1.1774737926193093,
        1.4983080316713226]

For PostLS2 configuration with a_comb = 31/32 Gllrf = 14 and TWCs at measured (with a tr of 0.65 to be sure)
df = [0,
      0]
G_tx = [1.1815415154561475,
        1.502249000727316]
        
For PostLS2 configuration with a_comb = 31/32 Gllrf = 16 and TWCs at measured
df = [0,
      0]
G_tx = [0.80,
        0.80]
        
For PostLS2 configuration with a_comb = 31/32 Gllrf = 20 and TWCs at measured
df = [0,
      0]
G_tx = [0.80,
        0.80]

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
plt.title('Power 3-section')
plt.plot(OTFB.OTFB_1.P_GEN[-h:])
plt.figure()
plt.title('Power 4-section')
plt.plot(OTFB.OTFB_2.P_GEN[-h:])

if BISECT:
    print(ant3, ant4)

if not BISECT:
    print()
    print('3-section power:', np.mean(OTFB.OTFB_1.P_GEN[-h:]))
    print('4-section power:', np.mean(OTFB.OTFB_2.P_GEN[-h:]))

if SHOW_PLT and not BISECT:
    plt.show()