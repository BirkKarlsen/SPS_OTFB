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

Vant3c = np.load('../data_files/with_impedance/Jan-21-2022/ch_2/1000turns_fwhm_288/sim_data/3sec_Vant_200.npy')
Vgen3c = np.load('../data_files/with_impedance/Jan-21-2022/ch_2/1000turns_fwhm_288/sim_data/3sec_Vindgen_200.npy')
Vbea3c = np.load('../data_files/with_impedance/Jan-21-2022/ch_2/1000turns_fwhm_288/sim_data/3sec_Vindbeam_200.npy')

Vant4c = np.load('../data_files/with_impedance/Jan-21-2022/ch_2/1000turns_fwhm_288/sim_data/4sec_Vant_200.npy')
Vgen4c = np.load('../data_files/with_impedance/Jan-21-2022/ch_2/1000turns_fwhm_288/sim_data/4sec_Vindgen_200.npy')
Vbea4c = np.load('../data_files/with_impedance/Jan-21-2022/ch_2/1000turns_fwhm_288/sim_data/4sec_Vindbeam_200.npy')


at.plot_IQ(Vant3, Vgen3, Vbea3, titstr='3-sec', norm=False, wind=3.5e6)
at.plot_IQ(Vant4, Vgen4, Vbea4, titstr='4-sec', norm=False, wind=3.5e6)
at.plot_IQ(Vant3c, Vgen3c, Vbea3c, titstr='3-sec corr', norm=False, wind=4e6)
at.plot_IQ(Vant4c, Vgen4c, Vbea4c, titstr='4-sec corr', norm=False, wind=4e6)
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