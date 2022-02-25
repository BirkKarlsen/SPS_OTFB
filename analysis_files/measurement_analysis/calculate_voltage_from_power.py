'''
This file was made to calculate the voltage giving rise to a given generator power consumption in order to reverse
engineer the voltage.

Author: Birk Emil Karlsen-Bæck
'''

# Imports
import numpy as np
from blond.llrf.impulse_response import SPS3Section200MHzTWC, SPS4Section200MHzTWC

TWC3 = SPS3Section200MHzTWC()
TWC4 = SPS4Section200MHzTWC()

# Parameters
omega_c = 2 * np.pi * 200394401.46888617        # [rad/s]
Z0 = 50                                         # [ohms]
tau3 = 4.619e-7                                 # [s]
tau4 = 6.207e-7                                 # [s]
domega3 = omega_c - TWC3.omega_r                # [rad/s]
domega4 = omega_c - TWC4.omega_r                # [rad/s]
Rg3 = 9850.907                                  # [ohms]
Rg4 = 13237.1566                                # [ohms]
n3 = 4
n4 = 2

# Input values for the power
power_3sec = 238e3                              # [W]
power_4sec = 391e3                              # [W]


# Functions
def igen_from_power(p, Z0):
    return np.sqrt(2 * p / Z0)

def vgen_from_igen(i, Rg, domega, tau):
    R_eff = 2 * (Rg / (domega * tau)) * np.sin(domega * tau / 2)
    return i * R_eff

# Calculation
i3 = igen_from_power(power_3sec, Z0)
i4 = igen_from_power(power_4sec, Z0)

V3 = vgen_from_igen(i3, Rg3, domega3, tau3)
V4 = vgen_from_igen(i4, Rg4, domega4, tau4)

print('Total voltage =', n3 * V3 + n4 * V4)
print('V_part =', n3 * V3 / (n3 * V3 + n4 * V4))

print('---- 3-section ----')
print('Igen =', i3)
print('Vant =', V3)

print('---- 4-section ----')
print('Igen =', i4)
print('Vant =', V4)


