'''
File to analyse the I/Q-signals and potential wells from simulations and from the measurements.

author: Birk Emil Karlsen-Bæck
'''

import utility_files.analysis_tools as at
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp



# I/Q-signals -----------------------------------------------------------------

# Importing the signals
Vant3 = np.load('../data_files/S1_GwI_fwhm_288_30000_Jan-02-2022/3sec_Vant_29999.npy')
Vgen3 = np.load('../data_files/S1_GwI_fwhm_288_30000_Jan-02-2022/3sec_Vindgen_29999.npy')
Vbea3 = np.load('../data_files/S1_GwI_fwhm_288_30000_Jan-02-2022/3sec_Vindbeam_29999.npy')

Vant4 = np.load('../data_files/S1_GwI_fwhm_288_30000_Jan-02-2022/4sec_Vant_29999.npy')
Vgen4 = np.load('../data_files/S1_GwI_fwhm_288_30000_Jan-02-2022/4sec_Vindgen_29999.npy')
Vbea4 = np.load('../data_files/S1_GwI_fwhm_288_30000_Jan-02-2022/4sec_Vindbeam_29999.npy')

at.plot_IQ(Vant3, Vgen3, Vbea3, titstr='3-sec', norm=False)
at.plot_IQ(Vant4, Vgen4, Vbea4, titstr='4-sec', norm=False)
plt.show()


# Potential Wells -------------------------------------------------------------

# Importing the antenna voltages
Vant3 = np.load('../data_files/S1_GwI_fwhm_288_30000_Jan-02-2022/3sec_Vant_29999.npy')
Vant4 = np.load('../data_files/S1_GwI_fwhm_288_30000_Jan-02-2022/4sec_Vant_29999.npy')

tot_Vant = 4 * Vant3 + 2 * Vant4

plt.figure()
plt.plot(tot_Vant.real, color='r', label='Real')
plt.plot(tot_Vant.imag, color='b', label='Imag')
plt.legend()

fig, ax = plt.subplots(2, 1, figsize=(8, 7))
ax[0].set_title('Abs')
ax[0].plot(np.abs(tot_Vant))

ax[1].set_title('Phase')
ax[1].plot(np.angle(tot_Vant))

plt.show()