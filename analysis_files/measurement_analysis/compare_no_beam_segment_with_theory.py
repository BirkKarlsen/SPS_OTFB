'''
File to plot the difference between the measured and theoretical power consumption of the SPS TWC Generator.

Author: Birk Emil Karlsen-Bæck
'''

# Imports --------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from analysis_files.measurement_analysis.import_data import get_power

plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 16
    })


# Plot -----------------------------------------------------------------------------------------------------------------

# Theoretical values for the power
P3 = [234e3, 185e3]         # 3-section [W]
P4 = [409e3, 324e3]         # 4-section [W]

# Measured Power
sec3_mean_tot, sec3_std_tot, sec4_mean_tot, sec4_std_tot = get_power()
dt = 8e-9
t = np.linspace(0, dt * 65536, 65536)
t_s = 1e6
P_s = 1e-3

fig, ax = plt.subplots(2, 1, figsize=(7, 8))

ax[0].set_title('3-section')
ax[0].plot(t * t_s, sec3_mean_tot * P_s, color='black', linestyle='--', label='M')
ax[0].fill_between(t * t_s, (sec3_mean_tot * 0.80) * P_s, (sec3_mean_tot * 1.20) * P_s, alpha=0.3, color='gray')

ax[0].hlines(P3[0] * P_s, t[0] * t_s, t[-1] * t_s, color='b', label='6.70 MV')
ax[0].hlines(P3[1] * P_s, t[0] * t_s, t[-1] * t_s, color='r', label='5.96 MV')

ax[0].set_xlim((0e-6 * t_s, 4e-6 * t_s))
ax[0].set_ylim((150e3 * P_s, 350e3 * P_s))

ax[0].set_ylabel(r'Power [kW]')
ax[0].set_xlabel(r'$\Delta t$ [$\mu$s]')



ax[1].set_title('4-section')
ax[1].plot(t * t_s, sec4_mean_tot * P_s, color='black', linestyle='--', label='M')
ax[1].fill_between(t * t_s, (sec4_mean_tot * 0.80) * P_s, (sec4_mean_tot * 1.20) * P_s, alpha=0.3, color='gray')

ax[1].hlines(P4[0] * P_s, t[0] * t_s, t[-1] * t_s, color='b', label='6.70 MV')
ax[1].hlines(P4[1] * P_s, t[0] * t_s, t[-1] * t_s, color='r', label='5.96 MV')

ax[1].set_xlim((0e-6 * t_s, 4e-6 * t_s))
ax[1].set_ylim((250e3 * P_s, 550e3 * P_s))

ax[1].set_ylabel(r'Power [kW]')
ax[1].set_xlabel(r'$\Delta t$ [$\mu$s]')


handles, labels = ax[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.01, 0.5))


plt.show()