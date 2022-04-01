'''
This file does the theoretical calculations for the case of a full SPS machine.

Author: Birk Emil Karlsen-Bæck
'''

# Import ----------------------------------------------------------------------
import numpy as np

# Functions -------------------------------------------------------------------
def generator_effective_impedance(f_r, f_c, R_gen, tau):
    r'''
    Calculates the effective impedance of the generator for a SPS machine.

    :param f_r: Resonant frequency of the TWC [Hz]
    :param f_c: Carrier frequency of the OTFB [Hz]
    :param R_gen: Generator resistance [\omega]
    :param tau: Cavity filling time [s]
    :return: R_eff
    '''
    domega = 2 * np.pi * (f_c - f_r)

    return 2 * (R_gen / (tau * domega)) * np.sin(domega * tau / 2)

def beam_effective_impedance(f_r, f_c, R_beam, tau):
    r'''
    Calculates the effective impedance of the beam for a SPS machine.

    :param f_r: Resonant frequency of the TWC [Hz]
    :param f_c: Carrier frequency of the OTFB [Hz]
    :param R_beam: Beam resistance [\omega]
    :param tau: Cavity filling time [s]
    :return: RI_eff, RQ_eff
    '''
    domega = 2 * np.pi * (f_c - f_r)

    RI_eff = - (2 * R_beam / (tau**2 * domega**2)) * (1 - np.cos(domega * tau))
    RQ_eff = (2 * R_beam / (tau * domega)) * (1 - np.sin(domega *tau) / (domega * tau))

    return RI_eff, RQ_eff


def theoretical_power(f_r, f_c, R_beam, R_gen, tau, I_beam, Vant, n_cav):
    '''
    Calculate the theoretical power consumption of the SPS generation assuming infinite gain and
    steady-state.

    :param f_r: Resonant frequency of the TWC [Hz]
    :param f_c: Carrier frequency of the OTFB [Hz]
    :param R_beam: Beam resistance [\omega]
    :param R_gen: Generator resistance [\omega]
    :param tau: Cavity filling time [s]
    :param I_beam: Beam current in IQ [A]
    :param Vant: Antenna voltage in IQ [V]
    :return: P_wo, P_wi - Power with and without beam-loading
    '''
    if type(f_r) is not list:
        f_r = [f_r, f_r]

    if type(R_beam) is not list:
        R_beam = [R_beam, R_beam]

    if type(R_gen) is not list:
        R_gen = [R_gen, R_gen]

    if type(tau) is not list:
        tau = [tau, tau]

    if type(Vant) is not list:
        Vant = [Vant, Vant]

    if type(n_cav) is not list:
        n_cav = [n_cav, n_cav]

    P_wo = np.zeros(2)
    P_wi = np.zeros(2)

    for i in range(2):
        Rg_eff = generator_effective_impedance(f_r[i], f_c, R_gen[i], tau[i])
        RbI_eff, RbQ_eff = beam_effective_impedance(f_r[i], f_c, R_beam[i], tau[i])

        # Calculation without beam
        II_gen_wo = 0 / Rg_eff
        IQ_gen_wo = Vant[i] / Rg_eff / n_cav[i]

        P_wo[i] = 25 * (II_gen_wo**2 + IQ_gen_wo**2)

        # Calculation with beam
        II_gen_wi = (-RbI_eff * n_cav[i] * I_beam) / Rg_eff / n_cav[i]
        IQ_gen_wi = (Vant[i] - RbQ_eff * n_cav[i] * I_beam) / Rg_eff / n_cav[i]

        P_wi[i] = 25 * (II_gen_wi**2 + IQ_gen_wi**2)

    return P_wo, P_wi


