'''
This is a simulation file simulating the exact configuration that was used for the HIRADMT 2 cycle during the start of
november 2021.

Author: Birk Emil Karlsen-Bæck
'''

# Argument Parser -------------------------------------------------------------
import argparse

parser = argparse.ArgumentParser(description="This file simulates the SPS OTFB with impedances.")

parser.add_argument("--n_turns", '-nt', type=int,
                    help="The number of turns to simulates, default is 1000")
parser.add_argument("--otfb_config", "-oc", type=int,
                    help="Different configurations of the OTFB parameters.")
parser.add_argument("--volt_config", "-vc", type=int,
                    help="Different values for the RF voltage.")
parser.add_argument("--save_dir", "-sd", type=str,
                    help="Name of directory to save the results to.")

args = parser.parse_args()


# Options for the Simulation --------------------------------------------------
GEN = False
SAVE_RESULTS = True
SINGLE_BATCH = False
LXPLUS = True
OTFB_CONFIG = 1
VOLT_CONFIG = 1
fit_type = 'fwhm'
mstdir = ''
dt_track = 1000
dt_ptrack = 10
dt_plot = 1000
dt_save = 1000


# Imports ---------------------------------------------------------------------
import matplotlib as mpl
#mpl.use('Agg')
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
from blond.impedances.impedance import TotalInducedVoltage, InducedVoltageFreq
from blond.impedances.impedance_sources import InputTable
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF

from SPS.impedance_scenario import scenario, impedance2blond


# Parameters ------------------------------------------------------------------
C = 2 * np.pi * 1100.009                        # Ring circumference [m]
gamma_t = 18.0                                  # Transition Gamma [-]
alpha = 1 / (gamma_t**2)                        # Momentum compaction factor [-]
p_s = 440e9                                     # Synchronous momentum [eV]
h = 4620                                        # 200 MHz harmonic number [-]
V = (0.911535 * 4 + 1.526871 * 2) * 1e6         # 200 MHz RF voltage [V]
phi = 0                                         # 200 MHz phase [-]

# Parameters for the SPS Cavity Feedback
V_part = 0.5442095845867135                     # Voltage partitioning [-]
G_tx = [0.1611031942822209,
        0.115855991237277]
G_llrf = 20

# Parameters for the SPS Impedance Model
freqRes = 43.3e3                                # Frequency resolution [Hz]
modelStr = "futurePostLS2_SPS_noMain200TWC.txt" # Name of Impedance Model

# Parameters for the Simulation
N_m = int(5e5)                                  # Number of macro-particles for tracking
N_t = 1000                                      # Number of turns to track
total_intensity = 3385.8196 * 10**10


# Change Simulation based on parsed arguments ---------------------------------
if args.n_turns is not None:
    N_t = args.n_turns

if args.otfb_config is not None:
    OTFB_CONFIG = args.otfb_config

if args.volt_config is not None:
    VOLT_CONFIG = args.volt_config

if args.save_dir is not None:
    mstdir = args.save_dir



if OTFB_CONFIG == 1:
    pass
elif OTFB_CONFIG == 2:
    rr = 0.9
    G_tx = [0.1611031942822209 * rr,
            0.115855991237277 * rr]
elif OTFB_CONFIG == 3:
    G_llrf = 16
    G_tx = [0.163212561182363,
            0.127838041632473]

if VOLT_CONFIG == 1:
    pass
elif VOLT_CONFIG == 2:
    V = 6660589.53641675
    V_part = 0.5517843967841601

# LXPLUS Simulation Configurations --------------------------------------------
if LXPLUS:
    lxdir = "/afs/cern.ch/work/b/bkarlsen/Simulation_Files/SPS_OTFB/"
else:
    lxdir = '../'
N_bunches = 288


# Objects ---------------------------------------------------------------------

# SPS Ring
ring = Ring(C, alpha, p_s, Proton(), n_turns=N_t)


# RF Station
rfstation = RFStation(ring, [h, 4 * h], [V, 0.19 * V], [0, np.pi], n_rf=2)


# Beam
bunch_intensities = np.load(lxdir + 'data_files/beam_parameters/avg_bunch_intensities_red.npy')
bunch_intensities = total_intensity * bunch_intensities / np.sum(bunch_intensities)  # normalize to 3385.8196 * 10**10
n_macro = N_m * N_bunches * bunch_intensities / np.sum(bunch_intensities)
beam = Beam(ring, int(np.sum(n_macro[:N_bunches])), int(total_intensity))


# Profile
profile = Profile(beam, CutOptions = CutOptions(cut_left=rfstation.t_rf[0,0] * (1000 - 2.5),
    cut_right=rfstation.t_rf[0,0] * (1000 + 72 * 5 * 4 + 250 * 3 + 125),
    n_slices=int(round(2**7 * (2.5 + 72 * 5 * 4 + 250 * 3 + 125)))))


# SPS Cavity Feedback
Commissioning = CavityFeedbackCommissioning(open_FF=True, debug=False, rot_IQ=1)
OTFB = SPSCavityFeedback(rfstation, beam, profile, post_LS2=True, V_part=V_part,
                         Commissioning=Commissioning, G_tx=G_tx, a_comb=31/32,
                         G_llrf=G_llrf)


# SPS Impedance Model
impScenario = scenario(modelStr)
impModel = impedance2blond(impScenario.table_impedance)

impFreq = InducedVoltageFreq(beam, profile, impModel.impedanceList, freqRes)
SPSimpedance_table = InputTable(impFreq.freq,impFreq.total_impedance.real*profile.bin_size,
                                impFreq.total_impedance.imag*profile.bin_size)
impedance_freq = InducedVoltageFreq(beam, profile, [SPSimpedance_table],
                                   frequency_resolution=freqRes)
total_imp = TotalInducedVoltage(beam, profile, [impedance_freq])


# Tracker Object without SPS OTFB
SPS_rf_tracker = RingAndRFTracker(rfstation, beam, TotalInducedVoltage=total_imp,
                                  CavityFeedback=None, Profile=profile)
SPS_tracker = FullRingAndRF([SPS_rf_tracker])


# Generate or import beam -----------------------------------------------------
if GEN:
    # Initialize the bunch
    bunch_lengths_fl = np.load(lxdir + 'data_files/beam_parameters/avg_bunch_length_full_length_red.npy')
    bunch_lengths_fwhm = np.load(lxdir + 'data_files/beam_parameters/avg_bunch_length_FWHM.npy')
    exponents = np.load(lxdir + 'data_files/beam_parameters/avg_exponent_red.npy')
    positions = np.load(lxdir + 'data_files/beam_parameters/position_fit.npy')

    bunch_length_list = bunch_lengths_fwhm * 1e-9

    distribution_options_list = {'bunch_length': bunch_length_list[:N_bunches],
                                 'type': 'binomial',
                                 'density_variable': 'Hamiltonian',
                                 'bunch_length_fit': 'fwhm',
                                 'exponent': exponents[:N_bunches]}

    bunch_positions = (positions - positions[0]) / rfstation.t_rf[0, 0]

    # If this fails, then generate without OTFB in the tracker and redefine the tracker after with OTFB.
    matched_from_distribution_density_multibunch(beam, ring, SPS_tracker, distribution_options_list,
                                                 N_bunches, bunch_positions[:N_bunches],
                                                 intensity_list=bunch_intensities[:N_bunches],
                                                 n_iterations=6, TotalInducedVoltage=total_imp)
    beam.dt += 1000 * rfstation.t_rf[0, 0]

    np.save(lxdir + f'data_files/with_impedance/generated_beams/generated_beam_{fit_type}_{N_bunches}_dE_r.npy',
            beam.dE)
    np.save(lxdir + f'data_files/with_impedance/generated_beams/generated_beam_{fit_type}_{N_bunches}_dt_r.npy',
            beam.dt)
else:
    beam.dE = np.load(
        lxdir + f'data_files/with_impedance/generated_beams/generated_beam_{fit_type}_{N_bunches}_dE_r.npy')
    beam.dt = np.load(
        lxdir + f'data_files/with_impedance/generated_beams/generated_beam_{fit_type}_{N_bunches}_dt_r.npy')


# Tracker Object with SPS Cavity Feedback -------------------------------------
SPS_rf_tracker = RingAndRFTracker(rfstation, beam, TotalInducedVoltage=total_imp,
                                  CavityFeedback=OTFB, Profile=profile, interpolation=True)
SPS_tracker = FullRingAndRF([SPS_rf_tracker])

profile.track()
total_imp.induced_voltage_sum()

# Set up directories for saving results ---------------------------------------
today = date.today()
sim_dir = f'data_files/with_impedance/{today.strftime("%b-%d-%Y")}/{mstdir}{N_t}turns_{fit_type}_{N_bunches}/'
sdir = lxdir + sim_dir + f'sim_data/'

if SAVE_RESULTS:
    np.save(lxdir + f'data_files/with_impedance/profile_data/generated_profile_{fit_type}_{N_bunches}_r', profile.n_macroparticles)
    np.save(lxdir + f'data_files/with_impedance/profile_data/generated_profile_bins_{fit_type}_{N_bunches}_r', profile.bin_centers)
    if not os.path.exists(lxdir + sim_dir + f'fig/'):
        os.makedirs(lxdir + sim_dir + f'fig/')

    if not os.path.exists(sdir):
        os.makedirs(lxdir + sim_dir + f'sim_data/')

# Show vital parameters for the simulation before tracking --------------------
print('----- Simulation Infomation -----')
print('Voltage:')
print('\tV =', V)
print('\tV_part', V_part)
print('One-Turn Feedback:')
print('\ta_comb =', OTFB.OTFB_1.a_comb, OTFB.OTFB_2.a_comb)
print('\tG_llrf =', OTFB.OTFB_1.G_llrf, OTFB.OTFB_2.G_llrf)

# Particle tracking -----------------------------------------------------------
if not GEN:
    fwhm_arr = np.zeros((N_bunches, N_t//dt_ptrack))
    pos_arr = np.zeros((N_bunches, N_t//dt_ptrack))
    pos_fit_arr = np.zeros((N_bunches, N_t//dt_ptrack))
    max_pow_arr = np.zeros((2, N_t//dt_ptrack))
    max_V_arr = np.zeros((2, N_t//dt_ptrack))
    int_arr = np.zeros(N_t//dt_ptrack)
    n = 0
    for i in range(N_t):
        SPS_tracker.track()
        profile.track()
        total_imp.induced_voltage_sum()
        OTFB.track()

        if i % dt_ptrack == 0:
            # Power
            OTFB.OTFB_1.calc_power()
            OTFB.OTFB_2.calc_power()

            fwhm_arr[:, n], pos_arr[:, n], pos_fit_arr[:, n], x_72, y_72 = dut.bunch_params(profile,
                                                                                            get_72=False)

            n += 1

        if i % dt_save == 0:
            dut.plot_params(fwhm_arr, pos_arr, pos_fit_arr,
                            max_pow_arr, max_V_arr, lxdir + sim_dir,
                            rfstation.t_rf[0,0], i, n - 1,
                            MB = not SINGLE_BATCH)

            dut.save_params(fwhm_arr, pos_arr, pos_fit_arr,
                            max_pow_arr, max_V_arr, lxdir + sim_dir)


        if i % dt_plot == 0:
            OTFB.OTFB_1.calc_power()
            OTFB.OTFB_2.calc_power()

            if SAVE_RESULTS:
                dut.save_plots_OTFB(OTFB, lxdir + sim_dir + f'fig/', i)
                dut.save_data(OTFB, lxdir + sim_dir + f'sim_data/', i)
                dut.save_profile(profile, lxdir + sim_dir + f'sim_data/', i)
                dut.plot_bbb_offset(pos_fit_arr[:72, n-1], lxdir + sim_dir + f'fig/', i)


    OTFB.OTFB_1.calc_power()
    OTFB.OTFB_2.calc_power()
    dut.save_plots_OTFB(OTFB, lxdir + sim_dir + f'fig/', N_t)
    dut.plot_bbb_offset(pos_fit_arr[:72, n - 1], lxdir + sim_dir + f'fig/', N_t)

    # Save the results to their respective directories
    if SAVE_RESULTS:
        dut.save_data(OTFB, lxdir + sim_dir + f'sim_data/', i)

        if not os.path.exists(lxdir + sim_dir + f'generated_beams/'):
            os.makedirs(lxdir + sim_dir + f'generated_beams/')
        np.save(lxdir + sim_dir + f'generated_beams/generated_beam_{fit_type}_{N_bunches}_dE_end_{N_t}.npy', beam.dE)
        np.save(lxdir + sim_dir + f'generated_beams/generated_beam_{fit_type}_{N_bunches}_dt_end_{N_t}.npy', beam.dt)

        if not os.path.exists(lxdir + sim_dir + f'profile_data/'):
            os.makedirs(lxdir + sim_dir + f'profile_data/')
        np.save(lxdir + sim_dir + f'profile_data/generated_profile_{fit_type}_{N_bunches}_end_{N_t}',
                profile.n_macroparticles)
        np.save(lxdir + sim_dir + f'profile_data/generated_profile_bins_{fit_type}_{N_bunches}_end_{N_t}',
                profile.bin_centers)

        np.save(lxdir + sim_dir + f"sim_data/induced_voltage", SPS_rf_tracker.totalInducedVoltage.induced_voltage)
        np.save(lxdir + sim_dir + f"sim_data/induced_voltage_time", SPS_rf_tracker.totalInducedVoltage.time_array)

