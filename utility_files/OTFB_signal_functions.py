r'''
This file contains a block-by-block function implementation of the OTFB.

author: Birk Emil Karlsen-Bæck
'''

# Imports ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from blond.llrf.signal_processing import comb_filter, cartesian_to_polar, \
    polar_to_cartesian, modulator, moving_average, \
    rf_beam_current
from blond.llrf.impulse_response import SPS3Section200MHzTWC, \
    SPS4Section200MHzTWC
from blond.utils import bmath as bm


# Functions -------------------------------------------------------------------

# LLRF Model
def set_point(V_part, rot_IQ, rf_voltage, phi_rf, n_coarse):
    V_set = polar_to_cartesian(
        V_part * rf_voltage,
        0.5 * np.pi - phi_rf + np.angle(rot_IQ))

    return V_set * np.ones(n_coarse)

def error_and_gain(G_llrf, V_SET, V_ANT):
    return G_llrf * (V_SET - V_ANT)

def comb(a_comb, DV_COMB_PREV, DV_GEN):
    return comb_filter(DV_COMB_PREV,
                       DV_GEN,
                       a_comb)

def one_turn_delay(n_delay, h, DV_COMB_OUT, PREV_DELAY):
    sig = np.concatenate(DV_COMB_OUT, PREV_DELAY)
    return sig[h - n_delay:-n_delay]

def mod_to_fr(omega_c, omega_r, t_rf, dphi_mod, dphi_rf, DV_DELAYED):
    return modulator(DV_DELAYED,
                    omega_c, omega_r, t_rf,
                    phi_0=dphi_mod + dphi_rf)

def mov_avg(n_mov_av, h, DV_MOD_FR, DV_MOD_PREV):
    DV_MOD_FR = np.concatenate(DV_MOD_PREV, DV_MOD_FR)
    return moving_average(DV_MOD_FR[-n_mov_av - h + 1:], n_mov_av)

def mod_to_frf(omega_c, omega_r, t_rf, dphi_mod, dphi_rf, DV_MOV_AVG):
    return modulator(DV_MOV_AVG,
                    omega_r, omega_c, t_rf,
                    phi_0=-(dphi_mod + dphi_rf))




def feedforward(sig, omega_c, omega_r, t_rf, tau, dphi_mod, dphi_rf, T_s, coeff_FF):
    n_coarse_FF = len(sig)
    n_FF = len(coeff_FF)
    n_FF_delay = int(0.5 * (n_FF - 1) + 0.5 * tau/t_rf/5)

    I_BEAM_COARSE_FF_MOD = np.zeros(2 * len(sig), dtype=complex)
    I_FF_CORR = np.zeros(2 * len(sig), dtype=complex)
    I_FF_CORR_MOD = np.zeros(2 * len(sig), dtype=complex)
    I_FF_CORR_DEL = np.zeros(2 * len(sig), dtype=complex)

    for i in range(2):
        # Do a down-modulation to the resonant frequency of the TWC
        I_BEAM_COARSE_FF_MOD[:n_coarse_FF] = I_BEAM_COARSE_FF_MOD[-n_coarse_FF:]
        I_BEAM_COARSE_FF_MOD[-n_coarse_FF:] = modulator(sig, omega_i=omega_c, omega_f=omega_r,
                                                        T_sampling=5 * T_s,
                                                        phi_0=(dphi_mod + dphi_rf))

        I_FF_CORR[:n_coarse_FF] = I_FF_CORR[-n_coarse_FF:]
        I_FF_CORR[-n_coarse_FF:] = np.zeros(n_coarse_FF)
        for ind in range(n_coarse_FF, 2 * n_coarse_FF):
            for k in range(n_FF):
                I_FF_CORR[ind] += coeff_FF[k] \
                                       * I_BEAM_COARSE_FF_MOD[ind - k]

        # Do a down-modulation to the resonant frequency of the TWC
        I_FF_CORR_MOD[:n_coarse_FF] = I_FF_CORR_MOD[-n_coarse_FF:]
        I_FF_CORR_MOD[-n_coarse_FF:] = modulator(I_FF_CORR[-n_coarse_FF:],
                                                           omega_i=omega_r, omega_f=omega_c,
                                                           T_sampling=5 * T_s,
                                                           phi_0=-(dphi_mod + dphi_rf))

        # Compensate for FIR filter delay
        I_FF_CORR_DEL[:n_coarse_FF] = I_FF_CORR_DEL[-n_coarse_FF:]
        I_FF_CORR_DEL[-n_coarse_FF:] = I_FF_CORR_MOD[n_FF_delay:n_FF_delay - n_coarse_FF]

    return I_FF_CORR_DEL
