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
    RQ_eff = (2 * R_beam / (tau * domega)) * (1 - np.sin(domega * tau) / (domega * tau))

    return RI_eff, RQ_eff


def theoretical_power(f_r, f_c, R_beam, R_gen, tau, I_beam, Vant, n_cav, VOLT=False, complex_antenna=False):
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
    Vg_wo = np.zeros(2, dtype=complex)
    Vg_wi = np.zeros(2, dtype=complex)
    Ig_wo = np.zeros(2, dtype=complex)
    Ig_wi = np.zeros(2, dtype=complex)
    Vb_wi = np.zeros(2, dtype=complex)

    for i in range(2):
        Rg_eff = generator_effective_impedance(f_r[i], f_c, R_gen[i], tau[i])
        RbI_eff, RbQ_eff = beam_effective_impedance(f_r[i], f_c, R_beam[i], tau[i])
        print('Rg_eff', Rg_eff, 'RbI_eff', RbI_eff, 'RbQ_eff', RbQ_eff)

        # Calculation without beam
        II_gen_wo = 0 / Rg_eff
        IQ_gen_wo = Vant[i] / Rg_eff / n_cav[i]
        if complex_antenna:
            II_gen_wo = Vant[i].real / Rg_eff / n_cav[i]
            IQ_gen_wo = Vant[i].imag / Rg_eff / n_cav[i]

        P_wo[i] = 25 * (II_gen_wo**2 + IQ_gen_wo**2)
        Vg_wo[i] = Rg_eff * n_cav[i] * (II_gen_wo + 1j * IQ_gen_wo)
        Ig_wo[i] = (II_gen_wo + 1j * IQ_gen_wo)

        # Calculation with beam
        II_gen_wi = (-RbI_eff * n_cav[i] * I_beam) / Rg_eff / n_cav[i]
        IQ_gen_wi = (Vant[i] - RbQ_eff * n_cav[i] * I_beam) / Rg_eff / n_cav[i]
        if complex_antenna:
            II_gen_wi = (Vant[i].real - RbI_eff * n_cav[i] * I_beam) / Rg_eff / n_cav[i]
            IQ_gen_wi = (Vant[i].imag - RbQ_eff * n_cav[i] * I_beam) / Rg_eff / n_cav[i]

        P_wi[i] = 25 * (II_gen_wi**2 + IQ_gen_wi**2)
        Vg_wi[i] = Rg_eff * n_cav[i] * (II_gen_wi + 1j * IQ_gen_wi)
        Ig_wi[i] = (II_gen_wi + 1j * IQ_gen_wi)
        Vb_wi[i] = n_cav[i] * (RbI_eff * I_beam + 1j * RbQ_eff * I_beam)

    if not VOLT:
        return P_wo, P_wi
    else:
        return P_wo, P_wi, Vg_wo, Vg_wi, Ig_wo, Ig_wi, Vb_wi


def theoretical_signals(O, I_beam):
    domega = O.omega_c - O.omega_r
    tau = O.TWC.tau
    R_g = O.TWC.R_gen
    R_b = O.TWC.R_beam

    a_gI = 2 * (R_g / tau / domega) * np.sin(domega * tau / 2)
    a_gQ = 2 * (R_g / tau / domega) * np.sin(domega * tau / 2)
    a_bI = -2 * (R_b / ((tau**2) * (domega**2))) * (1 - np.cos(domega * tau))
    a_bQ = 2 * (R_b / tau / domega) * (1 - (1/tau/domega) * np.sin(domega * tau))

    print('coefficients:',a_gI, a_gQ, a_bI, a_bQ)

    I_g_no_beam = ((1/a_gI) * np.real(O.V_set) + (1/a_gQ) * np.imag(O.V_set) * 1j) / O.n_cavities

    I_g_with_beam = ((1/a_gI) * (np.real(O.V_set) - a_bI * np.real(I_beam)) + (1/a_gQ) * (np.imag(O.V_set) - a_bQ * np.real(I_beam)) * 1j) / O.n_cavities

    return I_g_no_beam, I_g_with_beam


def theoretical_power_signal_cavity(V, f_c, n_sections):
    if n_sections == 3:
        # 3-section
        f_r = 200.038e6     # [Hz]
        R_beam = 485202     # [ohms]
        R_gen = 9851        # [ohms]
        tau = 462e-9        # [s]
        Vant = V            # [V]
        n_cav = 1
    else:
        # 4-section
        f_r = 199.995e6     # [Hz]
        R_beam = 876112     # [ohms]
        R_gen = 13237       # [ohms]
        tau = 621e-9        # [s]
        Vant = V            # [V]
        n_cav = 1

    P_wo, P_wi = theoretical_power(f_r, f_c, R_beam, R_gen, tau, 0, Vant, n_cav)

    return P_wo[0], P_wi[0]
