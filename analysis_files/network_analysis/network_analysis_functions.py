'''
Functions that are used in the network analysis.

Author: Birk Emil Karlsen-Bæck
'''

# Imports ---------------------------------------------------------------------
import numpy as np
import scipy.signal as sig

from blond.llrf.signal_processing import modulator

from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.llrf.cavity_feedback import SPSOneTurnFeedback, CavityFeedbackCommissioning

# Functions -------------------------------------------------------------------
def generate_sinusoid(omega, t, phase = 0, COMPLEX = True):
    if COMPLEX:
        return np.cos(omega * t + phase) + 1j * np.sin(omega * t + phase)
    else:
        return np.sin(omega * t + phase)

def convert_to_IQ(omega, omega_c, t):
    '''
    Generates an IQ signal on carrier frequency omega_c that is a sine-wave of frequency omega.

    :param omega: Frequency of sine-wave
    :param omega_c: Carrier Frequency
    :param t: time
    :return: IQ signal of the sine-wave
    '''
    sig = (1 - 1j * 0) * np.ones(len(t))
    T_s = t[1] - t[0]
    return modulator(sig, omega_i=omega_c, omega_f=omega, T_sampling=T_s)


def init_OTFB(df = 0):
    C = 2 * np.pi * 1100.009            # Ring circumference [m]
    gamma_t = 18.0                      # Gamma at transition
    alpha = 1 / gamma_t ** 2            # Momentum compaction factor
    p_s = 440e9                         # Synchronous momentum at injection [eV]
    h = 4620                            # 200 MHz system harmonic
    phi = 0.                            # 200 MHz RF phase

    # With this setting, amplitude in the two four-section, five-section
    # cavities must converge, respectively, to
    # 2.0 MV = 4.5 MV * 4/18 * 2
    # 2.5 MV = 4.5 MV * 5/18 * 2
    V = 6.7e6  # 200 MHz RF voltage

    N_t = 1  # Number of turns to track

    ring = Ring(C, alpha, p_s, Particle=Proton(), n_turns=N_t)
    rf = RFStation(ring, h, V, phi)

    N_m = 1e6  # Number of macro-particles for tracking
    N_b = 288 * 2.3e11  # Bunch intensity [ppb]

    # Gaussian beam profile
    beam = Beam(ring, N_m, N_b)

    profile = Profile(
        beam, CutOptions=CutOptions(
            cut_left=(-1.5) * rf.t_rf[0, 0],
            cut_right=(2.5) * rf.t_rf[0, 0],
            n_slices=4 * 64))

    Commissioning = CavityFeedbackCommissioning(open_FF=True, open_loop=True, open_drive=False,
                                                V_SET=1e6 * np.ones(2 * h),
                                                excitation=True)
    OTFB = SPSOneTurnFeedback(rf, beam, profile, n_sections=3, n_cavities=1, G_llrf=20, G_tx=1,
                              a_comb=31 / 32, df=df, V_part=1,
                              Commissioning=Commissioning)

    return OTFB

def calculate_freq_response_by_correlation(exc_in, exc_out):
    corr = sig.correlate
    return corr(exc_out, exc_in, 'valid') / corr(exc_in, exc_in, 'valid')

def calcualate_freq_response_vectorial(exc_in, exc_out, exc_freq):

    def get_demod_phasor(signal):
        signal_demod = modulator(signal, 2 * np.pi * exc_freq, 2 * np.pi / 4.990159369074305e-09,
                           T_sampling=4.990159369074305e-09)
        return np.mean(signal_demod)

    return get_demod_phasor(exc_out) / get_demod_phasor(exc_in)
