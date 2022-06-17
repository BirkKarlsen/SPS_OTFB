'''
File to check that the correct beam was generated.

Author: Birk Emil Karlsen-Bæck
'''


import numpy as np



part = np.load(f'../data_files/with_impedance/generated_beams/generated_beam_fwhm_288_100_dE_b.npy')


print(part.shape[0])