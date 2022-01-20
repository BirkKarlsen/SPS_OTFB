'''
File to make the induced voltages from the OTFB to match the corresponding calculation from impedances

author: Birk Emil Karlsen-Bæck
'''

# Imports
import numpy as np
from scipy.constants import e
import matplotlib.pyplot as plt
import logging

from blond.toolbox.logger import Logger
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import Profile, CutOptions
from blond.llrf.cavity_feedback import SPSOneTurnFeedback, \
    CavityFeedbackCommissioning
from blond.llrf.signal_processing import rf_beam_current
from blond.impedances.impedance_sources import TravelingWaveCavity
from blond.llrf.impulse_response import SPS3Section200MHzTWC, SPS4Section200MHzTWC
from blond.impedances.impedance import InducedVoltageTime, TotalInducedVoltage


# SPS machine parameters
# Machine and RF parameters
C = 2*np.pi*1100.009        # Ring circumference [m]
gamma_t = 18.0              # Gamma at transition
alpha = 1/gamma_t**2        # Momentum compaction factor
p_s = 25.92e9               # Synchronous momentum at injection [eV]
h = [4620]                  # 200 MHz system harmonic
V = [4.5e6]                 # 200 MHz RF voltage
phi = [0.]                  # 200 MHz RF phase
f_rf = 200.222e6            # Operational frequency of TWC, range ~200.1-200.36 MHz

# Beam and tracking parameters
N_m = 1e5                   # Number of macro-particles for tracking
N_b = 1.0e11                # Bunch intensity [ppb]
N_t = 1000                  # Number of turns to track


# Plot settings
plt.rc('axes', labelsize=12, labelweight='normal')
plt.rc('lines', linewidth=1.5, markersize=6)
plt.rc('font', family='sans-serif')
plt.rc('legend', fontsize=12)

LOGGING = False

# Logger for messages on console & in file
if LOGGING == True:
    Logger(debug = True)
else:
    Logger().disable()

# Set up machine parameters
ring = Ring(C, alpha, p_s, Particle=Proton(), n_turns=N_t)
logging.info("...... Machine parameters set!")

# Set up RF parameters
rf = RFStation(ring, h, V, phi, n_rf=1)
logging.info("...... RF parameters set!")

# Define beam and fill it
beam = Beam(ring, N_m, N_b)
bigaussian(ring, rf, beam, 3.2e-9/4, seed = 1234, reinsertion = True)
logging.info("...... Beam set!")
logging.info("Number of particles %d" %len(beam.dt))
logging.info("Time coordinates are in range %.4e to %.4e s" %(np.min(beam.dt),
                                                              np.max(beam.dt)))

profile = Profile(beam, CutOptions=CutOptions(cut_left=-1.e-9,
                                              cut_right=6.e-9, n_slices=100))
profile.track()








profile = Profile(beam, CutOptions=CutOptions(cut_left=-1.e-9,
                                              cut_right=6.e-9, n_slices=140))
profile.track()

# One-turn feedback around 3-, 4-, and 5-section cavities
omega_c = 2 * np.pi * f_rf
OTFB_3 = SPSOneTurnFeedback(rf, beam, profile, 3,
                            Commissioning=CavityFeedbackCommissioning(open_FF=True))
OTFB_4 = SPSOneTurnFeedback(rf, beam, profile, 4,
                            Commissioning=CavityFeedbackCommissioning(open_FF=True),
                            n_cavities=1, df=0.2275e6)
OTFB_5 = SPSOneTurnFeedback(rf, beam, profile, 5,
                            Commissioning=CavityFeedbackCommissioning(open_FF=True))
OTFB_3.counter = 0  # First turn
OTFB_4.counter = 0  # First turn
OTFB_5.counter = 0  # First turn
OTFB_3.omega_c = omega_c
OTFB_4.omega_c = omega_c
OTFB_5.omega_c = omega_c
# OTFB_3.TWC.impulse_response_beam(omega_c, profile.bin_centers)
# OTFB_4.TWC.impulse_response_beam(omega_c, profile.bin_centers)
# OTFB_5.TWC.impulse_response_beam(omega_c, profile.bin_centers)
OTFB_3.track()
OTFB_4.track()
OTFB_5.track()
V_ind_beam = OTFB_4.V_IND_FINE_BEAM[-OTFB_4.profile.n_slices:]
V_ind_beam = V_ind_beam

plt.figure()
convtime = np.linspace(-1e-9, -1e-9 + len(V_ind_beam.real) *
                       profile.bin_size, len(V_ind_beam.real))
plt.plot(convtime, V_ind_beam.real, 'b--')
plt.plot(convtime[:140], V_ind_beam.real[:140], 'b', label='Re(Vind), OTFB')
plt.plot(convtime, V_ind_beam.imag, 'r--')
plt.plot(convtime[:140], V_ind_beam.imag[:140], 'r', label='Im(Vind), OTFB')
plt.plot(convtime[:140], V_ind_beam.real[:140] * np.cos(OTFB_4.omega_c * convtime[:140]) \
         + V_ind_beam.imag[:140] * np.sin(OTFB_4.omega_c * convtime[:140]),
         color='purple', label='Total, OTFB')

# Comparison with impedances: FREQUENCY DOMAIN
TWC200_4 = TravelingWaveCavity(0.876e6, 200.222e6, 3.899e-6)
indVoltageTWC = InducedVoltageTime(beam, profile, [TWC200_4])
indVoltage = TotalInducedVoltage(beam, profile, [indVoltageTWC])
indVoltage.induced_voltage_sum()
plt.plot(indVoltage.time_array, indVoltage.induced_voltage, color='limegreen', label='Time domain w FFT')

# Comparison with impedances: TIME DOMAIN
TWC200_4.wake_calc(profile.bin_centers - profile.bin_centers[0])
wake1 = (TWC200_4.wake)
Vind = -profile.Beam.ratio * profile.Beam.Particle.charge * e * \
       np.convolve(wake1, profile.n_macroparticles, mode='full')[:140]
plt.plot(convtime[:140], Vind, color='teal', label='Time domain w conv')

# Wake from impulse response
OTFB_4.TWC.impulse_response_gen(omega_c, profile.bin_centers)
OTFB_5.TWC.impulse_response_gen(omega_c, profile.bin_centers)
OTFB_4.TWC.compute_wakes(profile.bin_centers)
OTFB_5.TWC.compute_wakes(profile.bin_centers)
wake2 = (OTFB_4.TWC.W_beam)
Vind = -profile.Beam.ratio * profile.Beam.Particle.charge * e * \
       np.convolve(wake2, profile.n_macroparticles, mode='full')[:140]
plt.plot(convtime[:140], Vind, color='turquoise', label='Wake, OTFB')
plt.xlabel("Time [s]")
plt.ylabel("Induced voltage [V]")
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.legend(loc=2)

plt.figure()
plt.plot(profile.bin_centers, wake1, label='from impedances')
plt.plot(profile.bin_centers, wake2, label='from OTFB')
plt.xlabel("Time [s]")
plt.ylabel("Wake field [Ohms/s]")
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.legend(loc=4)


plt.show()
logging.info("")
logging.info("Done!")

