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


def theoretical_power(f_r, f_c, R_beam, R_gen, tau, I_beam):

    if type(f_r) is not list:
        f_r = [f_r, f_r]

    if type(R_beam) is not list:
        R_beam = [R_beam, R_beam]

    if type(R_gen) is not list:
        R_gen = [R_gen, R_gen]

    if type(tau) is not list:
        tau = [tau, tau]

    for i in range(2):
        Rg_eff = generator_effective_impedance(f_r[i], f_c, R_gen[i], tau[i])
        RbI_eff, RbQ_eff = beam_effective_impedance(f_r[i], f_c, R_gen[i], tau[i])

        

