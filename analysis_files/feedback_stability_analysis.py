'''
This file was made to check the stability of the OTFB at the measured central frequencies.

Author: Birk Emil Karlsen-Bæck
'''

# Options ---------------------------------------------------------------------
PLT_3SEC_GEN = False
PLT_4SEC_GEN = False
PLT_3SEC_GEN_TIME = False
PLT_IMP = True


# Imports ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import utility_files.analysis_tools as at
from scipy.constants import c

from blond.llrf.signal_processing import TravellingWaveCavity
from blond.llrf.impulse_response import SPS3Section200MHzTWC, SPS4Section200MHzTWC
from blond.impedances.impedance_sources import TravelingWaveCavity


def generator_impedance(omega, omega_c, omega_r, tau):
    domega = omega_c - omega_r
    Omega = omega_c + omega_r
    tau = tau / 2

    output = np.sin(tau * (omega + domega)) / (tau * (omega + domega)) + \
             np.sin(tau * (omega - domega)) / (tau * (omega - domega))

    correction = np.sin(tau * (omega + Omega)) / (tau * (omega + Omega)) + \
                 np.sin(tau * (omega - Omega)) / (tau * (omega - Omega))

    first_order_correction = (2 / (tau * Omega)) * ((1 - omega / Omega) * np.sin(tau * (omega + Omega)) -
                                                    (1 + omega / Omega) * np.sin(tau * (omega - Omega)))

    return output, correction, first_order_correction

def generator_impulse_response(t, omega_c, omega_r, tau):
    domega = omega_c - omega_r
    Omega = omega_c + omega_r

    h_sg = (1 / tau) * at.rect(t / tau - 0.5) * (np.cos(domega * t) + np.cos(Omega * t))
    h_cg = (1 / tau) * at.rect(t / tau - 0.5) * (np.sin(domega * t) + np.sin(Omega * t))

    h_sg_u = (1 / tau) * at.rect(t / tau - 0.5) * np.cos(domega * t)
    h_cg_u = (1 / tau) * at.rect(t / tau - 0.5) * np.sin(domega * t)

    return h_sg, h_cg, h_sg_u, h_cg_u

def full_machine_theory(omega_c, omega_r, tau, Ib, Vant, Rg, Rb):
    domega = omega_c - omega_r

    Rg_eff = 2 * (Rg / (tau * domega)) * np.sin(domega * tau / 2)
    Rb_eff1 = -(2 * Rb / (tau**2 * domega**2)) * (1 - np.cos(domega * tau))
    Rb_eff2 = (2 * Rb / (tau * domega)) * (1 - (1 / (tau * domega)) * np.sin(tau * domega))

    VIb = Rb_eff1 * Ib
    VQb = Rb_eff2 * Ib

    IIg_wo = 0
    IQg_wo = Vant / Rg_eff

    IIg_wb = -VIb / Rg_eff
    IQg_wb = (Vant - VQb) / Rg_eff

    return IIg_wo, IQg_wo, IIg_wb, IQg_wb





# Parameters ------------------------------------------------------------------
omega_c = 2 * np.pi * 200394401.46888617
h = 4620
tau3 = 4.619183178616746e-07
omega_r3 = 2 * np.pi * 200.03766667e6
omega_r3_m = 2 * np.pi * 200.222e6
tau4 = 6.207027396266253e-07
omega_r4 = 2 * np.pi * 199.9945e6
Trev = h * 2 * np.pi / omega_c

n_omega = int(1e7)
omega = np.linspace(- 3 * omega_r3, 3 * omega_r3, n_omega)
f = omega / (2 * np.pi)

n_time = int(4620)
t = np.linspace(0, Trev, n_time)

# 3-section -------------------------------------------------------------------
Z_sg, Z_sg_corr, Z_sg_fo = generator_impedance(omega, omega_c, omega_r3, tau3)

if PLT_3SEC_GEN:
    plt.figure()
    plt.plot(f, Z_sg, label='Uncorr', color='black', linestyle='--')
    plt.plot(f, Z_sg + Z_sg_corr, label='Full', color='r')
    plt.plot(f, Z_sg + Z_sg_fo, label='FO', color='b')
    plt.legend()

    plt.figure()
    plt.plot(f, Z_sg_corr, label='Full', color='r')
    plt.plot(f, Z_sg_fo, label='FO', color='b')
    plt.legend()



# 4-section -------------------------------------------------------------------
Z_sg, Z_sg_corr, Z_sg_fo = generator_impedance(omega, omega_c, omega_r3, tau4)

if PLT_4SEC_GEN:
    plt.figure()
    plt.plot(f, Z_sg, label='Uncorr', color='black', linestyle='--')
    plt.plot(f, Z_sg + Z_sg_corr, label='Full', color='r')
    plt.plot(f, Z_sg + Z_sg_fo, label='FO', color='b')
    plt.legend()

    plt.figure()
    plt.plot(f, Z_sg_corr, label='Full', color='r')
    plt.plot(f, Z_sg_fo, label='FO', color='b')
    plt.legend()


# 3-section generator time-domain ---------------------------------------------
h_sg, h_cg, h_sg_u, h_cg_u = generator_impulse_response(t, omega_c, omega_r3, tau3)
h_sg_m, h_cg_m, h_sg_um, h_cg_um = generator_impulse_response(t, omega_c, omega_r3_m, tau3)


if PLT_3SEC_GEN_TIME:
    plt.figure()
    plt.plot(t, h_sg, color='r')
    plt.plot(t, h_cg, color='b')
    plt.plot(t, 2 * h_sg_u, color='r', linestyle='--')
    plt.plot(t, 2 * h_cg_u, color='b', linestyle='--')

    plt.figure()
    plt.plot(t, h_sg_m, color='r')
    plt.plot(t, h_cg_m, color='b')
    plt.plot(t, 2 * h_sg_um, color='r', linestyle='--')
    plt.plot(t, 2 * h_cg_um, color='b', linestyle='--')


# Full Machine Theory ---------------------------------------------------------
I_Ig_wo, I_Qg_wo, I_Ig_wb, I_Qg_wb = full_machine_theory(omega_c, omega_r3, tau3,
                                                         4 * 2.75, 3.675e6,
                                                         9850.907, 485201.87)
P_wo = 25 * (I_Ig_wo**2 + I_Qg_wo**2)
P_wb = 25 * (I_Ig_wb**2 + I_Qg_wb**2)
print('----- 3-section -----')
print('Power no beam:', P_wo / (4**2))
print('Power beam:', P_wb / (4**2))

I_Ig_wo, I_Qg_wo, I_Ig_wb, I_Qg_wb = full_machine_theory(omega_c, omega_r4, tau4,
                                                         2 * 2.75, 2.985e6,
                                                         13239.1566, 876111.578)
P_wo = 25 * (I_Ig_wo**2 + I_Qg_wo**2)
P_wb = 25 * (I_Ig_wb**2 + I_Qg_wb**2)
print('----- 4-section -----')
print('Power no beam:', P_wo / (2**2))
print('Power beam:', P_wb / (2**2))



# Impedance plot with revolution frequency ------------------------------------
l_cav = 32*0.374
v_g = 0.0946
tau = l_cav/(v_g*c)*(1 + v_g)
f_cav = omega_r3 / 2 / np.pi
n_cav = 4   # factor 2 because of two four/five-sections cavities
short_cavity = TravelingWaveCavity(l_cav**2 * n_cav * 27.1e3 / 8,
                                            f_cav, 2*np.pi*tau)
l_cav = 43*0.374
tau = l_cav/(v_g*c)*(1 + v_g)
n_cav = 2
f_cav = omega_r4 / 2 / np.pi
long_cavity = TravelingWaveCavity(l_cav**2 * n_cav * 27.1e3 / 8,
                                           f_cav, 2*np.pi*tau)
freq = np.linspace(0, 2 * f_cav, int(1e6))
long_cavity.imped_calc(freq)
short_cavity.imped_calc(freq)

TWC3 = short_cavity.impedance
TWC4 = long_cavity.impedance

if PLT_IMP:
    plt.figure()
    plt.title('3-section')
    plt.plot(freq, TWC3.real, color='r')
    plt.plot(freq, TWC3.imag, color='b')
    plt.grid()
    plt.vlines(200394401.46888617, -np.max(TWC3), np.max(TWC3), color='black')

    plt.figure()
    plt.title('4-section')
    plt.plot(freq, TWC4.real, color='r')
    plt.plot(freq, TWC4.imag, color='b')
    plt.grid()
    plt.vlines(200394401.46888617, -np.max(TWC4), np.max(TWC4), color='black')

    plt.figure()
    plt.title('Total Imepdance')
    plt.plot(freq, (TWC3 + TWC4).real, color='r')
    plt.plot(freq, (TWC3 + TWC4).imag, color='b')
    plt.grid()
    plt.vlines(200394401.46888617, -np.max(TWC3 + TWC4), np.max(TWC3 + TWC4), color='black')


plt.show()