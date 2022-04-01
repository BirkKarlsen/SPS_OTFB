'''
File to compute the theoretical power in the no beam segments of the HIRADMT2 November 2021
measurements.

Author: Birk Emil Karlsen-Bæck
'''

# Imports ---------------------------------------------------------------------
import numpy as np

import full_machine_theoretical_estimates as fmte


# Parameters ------------------------------------------------------------------
V = 6.70e6                                          # [V]
V_part = 0.5442095845867135                         # [-]
f_c = 1 / 4.990159369074305e-09                     # [Hz]

# Arrays
f_r = [0, 0]
R_beam = [0, 0]
R_gen = [0, 0]
tau = [0, 0]
Vant = [0, 0]
n_cav = [0, 0]

# 3-section
f_r[0] = 200.038e6                                  # [Hz]
R_beam[0] = 485202                                  # [ohms]
R_gen[0] = 9851                                     # [ohms]
tau[0] = 462e-9                                     # [s]
Vant[0] = V * V_part                                # [V]
n_cav[0] = 4

# 4-section
f_r[1] = 199.995e6                                  # [Hz]
R_beam[1] = 876112                                  # [ohms]
R_gen[1] = 13237                                    # [ohms]
tau[1] = 621e-9                                     # [s]
Vant[1] = V * (1 - V_part)                          # [V]
n_cav[1] = 2


# Calculation -----------------------------------------------------------------
P_wo, P_wi = fmte.theoretical_power(f_r, f_c, R_beam, R_gen, tau, 0, Vant, n_cav)


print(f'----- For V = {V/1e6:.2f} MV -----')
print(f'3-section power: {P_wo[0] / 1e3} kW')
print(f'4-section power: {P_wo[1] / 1e3} kW')


