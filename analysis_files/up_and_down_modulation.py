'''
File to test the up- and down-modulation function used in the OTFB model.

Author: Birk Emil Karlsen-Bæck
'''

# Import ----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from blond.llrf.signal_processing import modulator


# Parameters ------------------------------------------------------------------
f_i = 100
f_f = 200
omega_i = 2 * np.pi * f_i
omega_f = 2 * np.pi * f_f
T_n = 4620
T = 5 / f_i
T_sampling = T / T_n


# Signal ----------------------------------------------------------------------
t = np.linspace(0, T, T_n)

# The original signal
sin_sig_i = np.sin(omega_i * t)

# In I/Q
sin_sig_i_IQ = -np.ones(T_n) * 1j

# Modulate signal -------------------------------------------------------------
sin_sig_mod_IQ = modulator(sin_sig_i_IQ, omega_i, omega_f, T_sampling)

sin_sig_mod = np.abs(sin_sig_mod_IQ) * np.cos(omega_i * t + np.angle(sin_sig_mod_IQ))
sin_sig_f = np.sin(omega_f * t)

plt.figure()
plt.title('After Up-Modulation')
plt.plot(sin_sig_mod)
#plt.plot(sin_sig_i)
plt.plot(sin_sig_f)



sin_sig_mod_mod_IQ = modulator(sin_sig_mod_IQ, omega_f, omega_i, T_sampling)
sin_sig_mod_mod = np.abs(sin_sig_mod_mod_IQ) * np.cos(omega_i * t + np.angle(sin_sig_mod_mod_IQ))


plt.figure()
plt.title('After Down-Modulation')
plt.plot(sin_sig_mod_mod)
#plt.plot(sin_sig_i)
plt.plot(sin_sig_i)








plt.show()




