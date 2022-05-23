'''
File to plot the beam injected into the SPS flattop and how they compare with the bucket

Author: Birk Emil Karlsen-Bæck
'''

# Imports -------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os.path
from datetime import date
import utility_files.analysis_tools as at
import utility_files.data_utilities as dut

from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions_multibunch import matched_from_distribution_density_multibunch
from blond.llrf.cavity_feedback import SPSCavityFeedback, CavityFeedbackCommissioning
from blond.llrf.beam_feedback import BeamFeedback
from blond.impedances.impedance import TotalInducedVoltage, InducedVoltageFreq
from blond.impedances.impedance_sources import InputTable
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.trackers.utilities import separatrix

from SPS.impedance_scenario import scenario, impedance2blond

plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 16
    })

def clean_data(data, min_dt, max_dt, allowed_dE):
    dt = data['dt']
    dE = data['dE']

    invalid_particles = []

    print('Finding Invalid Particles...')
    for i in range(len(dt)):
        if dt[i] > max_dt or dt[i] < min_dt or np.abs(dE[i]) > allowed_dE:
            invalid_particles.append(i)

    print('Deleting Invalid Particles...')

    invalid_particles = np.array(invalid_particles)
    dt = np.delete(dt, invalid_particles)
    dE = np.delete(dE, invalid_particles)

    return {'dt': dt, 'dE': dE}

# Parameters ----------------------------------------------------------------------------------------------------------
C = 2 * np.pi * 1100.009                        # Ring circumference [m]
gamma_t = 18.0                                  # Transition Gamma [-]
alpha = 1 / (gamma_t**2)                        # Momentum compaction factor [-]
p_s = 440e9                                     # Synchronous momentum [eV]
h = 4620                                        # 200 MHz harmonic number [-]
V = (0.911535 * 4 + 1.526871 * 2) * 1e6         # 200 MHz RF voltage [V]
phi = 0                                         # 200 MHz phase [-]
V_800 = 0.19 * V                                # 800 MHz RF voltage [V]

# Parameters for the SPS Cavity Feedback
V_part = 0.5442095845867135                     # Voltage partitioning [-]
G_tx = [1,
        1]
G_llrf = 20
df = [0,
      0]
G_ff = 1

# Parameters for the SPS Impedance Model
freqRes = 43.3e3                                # Frequency resolution [Hz]
modelStr = "futurePostLS2_SPS_noMain200TWC.txt" # Name of Impedance Model

# Parameters for the Simulation
N_m = int(5e5)                                  # Number of macro-particles for tracking
N_t = 1000                                      # Number of turns to track
N_ir = 0                                        # Number of turns for the intensity ramp

N_tot = N_t + N_ir
total_intensity = 3385.8196e10
lxdir = '../'
N_bunches = 288
fit_type = 'fwhm'
bunch_length_factor = 1.0


# Objects -------------------------------------------------------------------------------------------------------------

# SPS Ring
ring = Ring(C, alpha, p_s, Proton(), n_turns=N_tot)


# RF Station
rfstation = RFStation(ring, [h, 4 * h], [V, V_800], [0, np.pi], n_rf=2)
rfstation_ve89 = RFStation(ring, [h, 4 * h], [V * 0.89, V_800], [0, np.pi], n_rf=2)

# Beam
bunch_intensities = np.load(lxdir + 'data_files/beam_parameters/avg_bunch_intensities_red.npy')
bunch_intensities = total_intensity * bunch_intensities / np.sum(bunch_intensities)  # normalize to 3385.8196 * 10**10
n_macro = N_m * N_bunches * bunch_intensities / np.sum(bunch_intensities)
beam = Beam(ring, int(np.sum(n_macro[:N_bunches])), int(total_intensity))

# Profile
profile = Profile(beam, CutOptions = CutOptions(cut_left=rfstation.t_rf[0,0] * (1000 - 2.5),
        cut_right=rfstation.t_rf[0,0] * (1000 + 72 * 5 * 4 + 250 * 3 + 125),
        n_slices=int(round(2**7 * (2.5 + 72 * 5 * 4 + 250 * 3 + 125)))))


# SPS OTFB
Commissioning = CavityFeedbackCommissioning(open_FF=not False, debug=False, rot_IQ=1,
                                            FIR_filter=1)
OTFB = SPSCavityFeedback(rfstation, beam, profile, post_LS2=True, V_part=V_part,
                         Commissioning=Commissioning, G_tx=G_tx, a_comb=31/32,
                         G_llrf=G_llrf, df=df, G_ff=G_ff)

beam.dE = np.load(
        lxdir + f'data_files/with_impedance/generated_beams/'
                f'generated_beam_{fit_type}_{N_bunches}_{100 * bunch_length_factor:.0f}_dE_f.npy')
beam.dt = np.load(
        lxdir + f'data_files/with_impedance/generated_beams/'
                f'generated_beam_{fit_type}_{N_bunches}_{100 * bunch_length_factor:.0f}_dt_f.npy')


# Tracker Object with SPS Cavity Feedback -----------------------------------------------------------------------------
SPS_rf_tracker = RingAndRFTracker(rfstation, beam, TotalInducedVoltage=None,
                                  CavityFeedback=OTFB, Profile=profile, BeamFeedback=None,
                                  interpolation=True)
SPS_tracker = FullRingAndRF([SPS_rf_tracker])

profile.track()


# Plot Longitudinal Phase-space ---------------------------------------------------------------------------------------
import seaborn as sns



data = {'dE': beam.dE, 'dt': beam.dt}
print(type(data), list(data.keys()), len(data['dt']), len(data['dE']))

dt_plot = 1
Delta_dt = 1 * rfstation.t_rf[0,0]
separatrix_res = 1000
air_over_bucket = 1.2

fwhm_arr, pos_arr, pos_fit_ar, x_72, y_72 = dut.bunch_params(profile, get_72=False)

for i in range(N_bunches):
    print(f'Bunch {i}')
    min_dt = pos_fit_ar[i] - Delta_dt
    max_dt = pos_fit_ar[i] + Delta_dt

    dts = np.linspace(min_dt, max_dt, separatrix_res)
    des = separatrix(ring, rfstation, dts)
    des_ve89 = separatrix(ring, rfstation_ve89, dts)

    data_i = clean_data(data, min_dt=min_dt, max_dt=max_dt, allowed_dE=np.max(des) * air_over_bucket)

    print(type(data_i), list(data_i.keys()), len(data_i['dt']), len(data_i['dE']))

    cp = sns.color_palette('coolwarm', as_cmap=True)
    sns.displot(data_i, x='dt', y='dE', cbar=True, cmap=cp, vmin=0, vmax=120)
    plt.title(f'Bunch {i}')
    plt.xlabel(r'$\Delta t$ [ns]')
    plt.ylabel(r'$\Delta E$ [MeV]')
    plt.xlim((min_dt, max_dt))
    plt.ylim((-np.max(des) * air_over_bucket, np.max(des) * air_over_bucket))

    plt.plot(dts, des, color='black')
    plt.plot(dts, -des, color='black')
    plt.plot(dts, des_ve89, color='black', linestyle='--')
    plt.plot(dts, -des_ve89, color='black', linestyle='--')

    if i % dt_plot == 0:
        plt.show()