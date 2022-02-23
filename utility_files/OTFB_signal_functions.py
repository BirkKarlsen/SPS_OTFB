r'''
This file contains a block-by-block function implementation of the OTFB.

author: Birk Emil Karlsen-Bæck
'''

# Imports ---------------------------------------------------------------------
import numpy as np

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






