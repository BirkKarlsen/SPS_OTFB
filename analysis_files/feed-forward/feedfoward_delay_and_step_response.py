'''
File to test the FF delay and how it responds to a step function.

Author: Birk Emil Karlsen-Bæck
'''


# Imports ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import utility_files.OTFB_signal_functions as osf
from scipy.constants import c

from blond.llrf.signal_processing import feedforward_filter_TWC3_1, feedforward_filter_TWC3_2, \
    feedforward_filter_TWC3_3
from blond.llrf.impulse_response import SPS3Section200MHzTWC
from blond.impedances.impedance_sources import TravelingWaveCavity


# Parameters ------------------------------------------------------------------
h = 4620                                    # [-]
omega_c = 1259115158.950353                 # [rad/s]
omega_r = 1258031928.574111                 # [rad/s]
t_rf = 4.990159369074305e-09                # [s]
tau = 4.619183178616746e-07                 # [s]
T_s = 4.990159369074305e-09                 # [s]
dphi_mod = 0                                # [rad]
dphi_rf = 0                                 # [rad]
df = 0.18433333e6
omega_r -= 2 * np.pi * df

# Make a step function --------------------------------------------------------
step_func = np.zeros(h // 5)
step_func[-h//10:] = 1
t_fine = np.linspace(0, T_s * (h-1), h)
t_coarse = t_fine[::5]
print(omega_r / (2 * np.pi))

plt.figure()
plt.plot(step_func)

# Use feed-forward model ------------------------------------------------------
output = osf.feedforward(step_func, omega_c, omega_c, t_rf, tau, dphi_mod, dphi_rf,
                         T_s, coeff_FF=feedforward_filter_TWC3_1)

# Result from Beam induced voltage
TWC3 = SPS3Section200MHzTWC(df=0)
TWC3.impulse_response_beam(omega_c, t_fine, t_coarse)

TWC3.impulse_response_gen(omega_c, t_fine)
gen_h_FF = TWC3.h_gen[::5]

TWC3.impulse_response_gen(omega_c, t_coarse)
gen_h_FF2 = TWC3.h_gen

TWC3.impulse_response_beam(omega_c, t_fine, t_coarse)

beam_ind = np.convolve(step_func, TWC3.h_beam_coarse)[:len(step_func)]
beam_ind = beam_ind[-h//5:]

gen_ind_FF = np.convolve(output, gen_h_FF)[:len(output)]
gen_ind_FF = gen_ind_FF[-h//5:]

gen_ind_FF2 = np.convolve(output, gen_h_FF2)[:len(output)]
gen_ind_FF2 = gen_ind_FF2[-h//5:]



plt.figure()
plt.plot(-beam_ind.real / (2*TWC3.R_beam/TWC3.tau))
plt.plot(gen_ind_FF2.real / (TWC3.R_gen / TWC3.tau) * 10)

#plt.figure()
#plt.plot(beam_ind.imag)
#plt.plot(gen_ind_FF.imag * 750)


# From impedance model
# Cavities
l_cav = 32 * 0.374
v_g = 0.0946
tau = l_cav / (v_g * c) * (1 + v_g)
f_cav = 200.222e6
n_cav = 1  # factor 2 because of two four/five-sections cavities
short_cavity = TravelingWaveCavity(l_cav ** 2 * n_cav * 27.1e3 / 8,
                                       f_cav, 2 * np.pi * tau)

short_cavity.wake_calc(t_coarse)
beam_ind = np.convolve(step_func, short_cavity.wake)

plt.figure()
plt.plot(beam_ind.real)
plt.plot(beam_ind.imag)

plt.show()
