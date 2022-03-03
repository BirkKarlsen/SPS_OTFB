'''
This file was made to analyse power related results from the SPS HIRADMT2 simulations.

Author: Birk Emil Karlsen-Bæck
'''

# Imports ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt


# Configurations for the file -------------------------------------------------
data_dir = f'../data_files/hiradmt2_nov_2021_sims/'
sim_dir = f'tr90_np/'
part_of_turn = 1000

PLT_POWER = False

# Get data from files ---------------------------------------------------------
power3 = np.load(data_dir + sim_dir + '3sec_power_19999.npy')
power4 = np.load(data_dir + sim_dir + '4sec_power_19999.npy')
vant3 = np.load(data_dir + sim_dir + '3sec_Vant_19999.npy')
vant4 = np.load(data_dir + sim_dir + '4sec_Vant_19999.npy')

# Analyse data ----------------------------------------------------------------
Vant = np.mean(np.abs(vant3[-part_of_turn:]) + np.abs(vant4[-part_of_turn:])) * 1e-6
power_3sec = np.mean(power3[-part_of_turn:]) * 1e-3
power_4sec = np.mean(power4[-part_of_turn:]) * 1e-3

# Print data ------------------------------------------------------------------
print('----- Data Analysis -----')
print(f'\tAntenna Voltage: {Vant} MV')
print(f'\tPower 3-section: {power_3sec} kW')
print(f'\tPower 4-section: {power_4sec} kW')

# Plot data -------------------------------------------------------------------

if PLT_POWER:
    plt.figure()
    plt.title('Power 3-section')
    plt.plot(power3)

    plt.figure()
    plt.title('Power 3-section')
    plt.plot(power4)

plt.show()
