'''
File to check that the correct beam was generated.

Author: Birk Emil Karlsen-Bæck
'''


import numpy as np
import matplotlib.pyplot as plt
import utility_files.data_utilities as dut
import utility_files.analysis_tools as at


from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring


part_dE = np.load(f'../data_files/with_impedance/generated_beams/generated_beam_fwhm_288_110_dE_f.npy')
part_dt = np.load(f'../data_files/with_impedance/generated_beams/generated_beam_fwhm_288_110_dt_f.npy')


# Parameters ------------------------------------------------------------------
C = 2 * np.pi * 1100.009                        # Ring circumference [m]
gamma_t = 18.0                                  # Transition Gamma [-]
alpha = 1 / (gamma_t**2)                        # Momentum compaction factor [-]
p_s = 440e9                                     # Synchronous momentum [eV]
h = 4620                                        # 200 MHz harmonic number [-]
V = (0.911535 * 4 + 1.526871 * 2) * 1e6         # 200 MHz RF voltage [V]
phi = 0                                         # 200 MHz phase [-]


N_bunches = 288
total_intensity = 3385.8196e10
N_m = int(5e5)


# Objects ---------------------------------------------------------------------

# SPS Ring
ring = Ring(C, alpha, p_s, Proton(), n_turns=1)


# RF Station
rfstation = RFStation(ring, [h, 4 * h], [V, 0.19 * V], [0, np.pi], n_rf=2)

bunch_intensities = np.load('../data_files/beam_parameters/avg_bunch_intensities_red.npy')
bunch_intensities = total_intensity * bunch_intensities / np.sum(bunch_intensities)  # normalize to 3385.8196 * 10**10
n_macro = N_m * N_bunches * bunch_intensities / np.sum(bunch_intensities)
beam = Beam(ring, int(np.sum(n_macro[:N_bunches])), int(total_intensity))

beam.dE = part_dE
beam.dt = part_dt

profile = Profile(beam, CutOptions = CutOptions(cut_left=rfstation.t_rf[0,0] * (1000 - 2.5),
        cut_right=rfstation.t_rf[0,0] * (1000 + 72 * 5 * 4 + 250 * 3 + 125),
        n_slices=int(round(2**7 * (2.5 + 72 * 5 * 4 + 250 * 3 + 125)))))
profile.track()


fwhm_arr, pos_arr, pos_fit_arr, x_72, y_72 = dut.bunch_params(profile, get_72=False)

dut.plot_bbb_offset(pos_fit_arr, 4, '', 0, show=True)
