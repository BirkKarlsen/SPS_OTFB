'''
File to test the FF delay and how it responds to a step function.

Author: Birk Emil Karlsen-Bæck
'''


# Imports ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import utility_files.OTFB_signal_functions as osf

from blond.llrf.signal_processing import feedforward_filter_TWC3_1, feedforward_filter_TWC3_2, \
    feedforward_filter_TWC3_3


# Parameters ------------------------------------------------------------------
h = 4620                                    # [-]
omega_c = 1259115158.950353                 # [rad/s]
omega_r = 1258031928.574111                 # [rad/s]
t_rf = 4.990159369074305e-09                # [s]
tau = 4.619183178616746e-07                 # [s]
T_s = 4.990159369074305e-09                 # [s]
dphi_mod = 0                                # [rad]
dphi_rf = 0                                 # [rad]

# Make a step function --------------------------------------------------------
step_func = np.zeros(h // 5)
step_func[-h//10:] = 1

plt.figure()
plt.plot(step_func)

# Use feed-forward model ------------------------------------------------------
output = osf.feedforward(step_func, omega_c, omega_r, t_rf, tau, dphi_mod, dphi_rf,
                         T_s, coeff_FF=feedforward_filter_TWC3_3)

plt.plot(output.real[-h//5:])
plt.plot(output.imag[-h//5:])

print(np.sum(feedforward_filter_TWC3_1))
print(np.sum(feedforward_filter_TWC3_2))
print(np.sum(feedforward_filter_TWC3_3))






plt.show()
