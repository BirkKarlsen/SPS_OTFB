'''
This file simulates the bunches with the impedances of the SPS.

author: Birk Emil Karlsen-Bæck
'''

import argparse

parser = argparse.ArgumentParser(description="This file simulates the SPS OTFB with impedances.")

parser.add_argument("--n_turns", '-nt', type=int,
                    help="The number of turns to simulates, default is 1000")
parser.add_argument("--n_plot", "-np", type=int,
                    help="The number of turns between each plot, default is 1000")
parser.add_argument("--sgl_bth", "-sb", type=int,
                    help="Single batch (1) or four batches (0), default is 0")
parser.add_argument("--generate", "-g", type=int,
                    help="Generate beam (1) or run simulation (0), default is 0")
parser.add_argument("--impedances", "-imp", type=int,
                    help="Include impedances (1) or not (0), default is 1")
parser.add_argument("--master_directory", "-md", type=str,
                    help="Option to save to a new folder within the with_impedance-folder")
parser.add_argument("--alpha_comb", "-ac", type=float,
                    help="The alpha coefficient for the comb filter, default is 63/64")
parser.add_argument("--llrf_gain", "-lg", type=float,
                    help="The LLRF gain, default is 10")
parser.add_argument("--tx_gain", "-tg", type=float,
                    help="The transmitter gain, default is optimized for pre-tracking")
parser.add_argument("--study_osc", "-so", type=int,
                    help="Option to plot and save the possible oscillation of the 3-sec power, default is False (0)")
parser.add_argument("--n_save", "-ns", type=int,
                    help="Option to change the number of turns between saving and plotting of parameters, default is 10000")
parser.add_argument("--n_ptrack", "-npt", type=int,
                    help="Option to change the number of turns between tracking of parameters, default is 10")
parser.add_argument("--central_freq", '-cf', type=int,
                    help="Option to choose the central frequnecy of the 4-section, (1) 200.1, (2) 200.222, or (3) 199.995")
parser.add_argument("--rot_iq", "-ri", type=int,
                    help="Option to rotate the IQ-signals by 180 degrees (-1), default is 1")
parser.add_argument("--dphase", "-dp", type=float,
                    help="Option to offset the total phase of the RF system in the simulation, default is 0.")



args = parser.parse_args()


# Imports -------------------------------------------------------------------------------------------------------------

print('Importing...\n')
import matplotlib as mpl
mpl.use('Agg') # TODO: uncomment
import matplotlib.pyplot as plt
import numpy as np
import utility_files.data_utilities as dut
import os.path
from datetime import date

from blond.llrf.cavity_feedback import SPSCavityFeedback, CavityFeedbackCommissioning
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions_multibunch import matched_from_distribution_density_multibunch
from blond.trackers.tracker import FullRingAndRF, RingAndRFTracker
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from blond.impedances.impedance_sources import InputTable

# SPS Impedance
from SPS.impedance_scenario import scenario, impedance2blond

# TODO: change back - DONE
fit_type = 'fwhm'
SINGLE_BATCH = False
GENERATE = False                           # TODO: True
SAVE_RESULTS = True
LXPLUS = True                              # TODO: change back before copy to lxplus
SPS_IMP = True
STDY_OSC = False
TRACK_IMP = True
OMEGA_SCENARIO = 3
ROT_IQ = 1

if not LXPLUS:
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 16
    })


# Parameters ----------------------------------------------------------------------------------------------------------
C = 2 * np.pi * 1100.009                        # Ring circumference [m]
gamma_t = 18.0                                  # Transition Gamma [-]
alpha = 1 / (gamma_t**2)                        # Momentum compaction factor [-]
p_s = 440e9                                     # Synchronous momentum [eV]
h = 4620                                        # 200 MHz harmonic number [-]
V = (0.911535 * 4 + 1.526871 * 2) * 1e6         # 200 MHz RF voltage [V]
phi = 0                                         # 200 MHz phase [-]


# Parameters for the Simulation
N_m = int(5e5)                                  # Number of macro-particles for tracking
N_t = 100                                      # Number of turns to track

dt_plot = 1000 # TODO: 1000
dt_track = 1000 # TODO: 1000
dt_save = 1000 # TODO: 1000
dt_ptrack = 10


# Arguments parsed
if args.n_turns is not None:
    N_t = args.n_turns
if args.n_plot is not None:
    dt_plot = args.n_plot
if args.sgl_bth is not None:
    SINGLE_BATCH = bool(args.sgl_bth)
if args.generate is not None:
    GENERATE = bool(args.generate)
if args.impedances is not None:
    SPS_IMP = bool(args.impedances)

if args.master_directory is not None:
    mstdir = args.master_directory
else:
    mstdir = ""

if args.alpha_comb is not None:
    a_comb = args.alpha_comb
else:
    a_comb = 63/64

if args.llrf_gain is not None:
    llrf_g = args.llrf_gain
else:
    llrf_g = 20

if args.tx_gain is not None:
    tx_g = args.tx_gain
else:
    tx_g = [0.2564371551236985, 0.53055789217211453]

if args.study_osc is not None:
    STDY_OSC = bool(args.study_osc)
if args.n_save is not None:
    dt_save = args.n_save
if args.n_ptrack is not None:
    dt_ptrack = args.n_ptrack

if args.central_freq is not None:
    OMEGA_SCENARIO = int(args.central_freq)

if SINGLE_BATCH:
    N_bunches = 72                              # Number of bunches
else:
    N_bunches = 288                             # Number of bunches

if args.rot_iq is not None:
    ROT_IQ = args.rot_iq

if args.dphase is not None:
    dphase = args.dphase
else:
    dphase = 0

print('Fit type:', fit_type)
print('Number of Bunches:', N_bunches)
print('GENERATE:', GENERATE)
print('SPS_IMP:', SPS_IMP)
print("SAVE_RESULTS:", SAVE_RESULTS)
print("LXPLUS:", LXPLUS)
print("STDY_OSC:", STDY_OSC)
print("OMEGA_SCENARIO:", OMEGA_SCENARIO)
print()


if LXPLUS:
    lxdir = "/afs/cern.ch/work/b/bkarlsen/Simulation_Files/SPS_OTFB/"
else:
    lxdir = "../"

if OMEGA_SCENARIO == 1:
    domega = [0, 0.1055e6]
elif OMEGA_SCENARIO == 2:
    domega = [0, 0.2275e6]
else:
    domega = [0, 0]


# Objects -------------------------------------------------------------------------------------------------------------
print('Initializing Objects...\n')


# Ring
SPS_ring = Ring(C, alpha, p_s, Proton(), N_t)

# RFStation
rfstation = RFStation(SPS_ring, [h, 4 * h], [V, 0.19 * V], [0 + dphase, np.pi + 4 * dphase], n_rf=2)
# TODO: Shift with pi/2 only main harm. in fourth harmonic add delta-phi * 4

# SINGLE BUNCH FIRST
# Beam
bunch_intensities = np.load(lxdir + 'data_files/beam_parameters/avg_bunch_intensities_red.npy')
total_intensity = 3385.8196 * 10**10

bunch_intensities = total_intensity * bunch_intensities / np.sum(bunch_intensities)  # normalize to 3385.8196 * 10**10
n_macro = N_m * N_bunches * bunch_intensities / np.sum(bunch_intensities)

beam = Beam(SPS_ring, int(np.sum(n_macro[:N_bunches])), int(total_intensity))

# Profile
profile = Profile(beam, CutOptions = CutOptions(cut_left=0.e-9,
    cut_right=rfstation.t_rev[0], n_slices=2**7 * 4620))

# One Turn Feedback
V_part = 0.5442095845867135
# TODO: Run with Gtx of 1

#G_tx_ls = [0.2712028956, 0.58279606]
#G_llrf_ls = [41.751786, 35.24865]
#llrf_g = G_llrf_ls


Commissioning = CavityFeedbackCommissioning(open_FF=True, debug=False,
                                            rot_IQ=ROT_IQ)
OTFB = SPSCavityFeedback(rfstation, beam, profile, post_LS2=True, V_part=V_part,
                         Commissioning=Commissioning, G_tx=tx_g, a_comb=a_comb,
                         G_llrf=llrf_g, df=domega)   # TODO: change back to only 20

# Impedance of the SPS
if SPS_IMP:
    freqRes = 43.3e3          # Hz

    modelStr = "futurePostLS2_SPS_noMain200TWC.txt"
    impScenario = scenario(modelStr)
    impModel = impedance2blond(impScenario.table_impedance)

    impFreq = InducedVoltageFreq(beam, profile, impModel.impedanceList,
                                 freqRes)

    SPSimpedance_table = InputTable(impFreq.freq,impFreq.total_impedance.real*profile.bin_size,
                                    impFreq.total_impedance.imag*profile.bin_size)
    impedance_freq = InducedVoltageFreq(beam, profile, [SPSimpedance_table],
                                   frequency_resolution=freqRes)
    total_imp = TotalInducedVoltage(beam, profile, [impedance_freq])

else:
    total_imp = None

# Tracker object for full ring
SPS_rf_tracker = RingAndRFTracker(rfstation, beam, TotalInducedVoltage=total_imp,
                                  CavityFeedback=None, Profile=profile)
SPS_tracker = FullRingAndRF([SPS_rf_tracker])


# Initialize the bunch
bunch_lengths_fl = np.load(lxdir + 'data_files/beam_parameters/avg_bunch_length_full_length_red.npy')
bunch_lengths_fwhm = np.load(lxdir + 'data_files/beam_parameters/avg_bunch_length_FWHM.npy')
exponents = np.load(lxdir + 'data_files/beam_parameters/avg_exponent_red.npy')
positions = np.load(lxdir + 'data_files/beam_parameters/avg_positions_red.npy')

if fit_type == 'fwhm':
    bunch_length_list = bunch_lengths_fwhm * 1e-9
else:
    bunch_length_list = bunch_lengths_fl * 1e-9


distribution_options_list = {'bunch_length': bunch_length_list[:N_bunches],
                              'type': 'binomial',
                              'density_variable': 'Hamiltonian',
                              'bunch_length_fit': fit_type,
                              'exponent': exponents[:N_bunches]}

bunch_positions = (positions - positions[0]) / rfstation.t_rf[0,0]

if GENERATE:
    # If this fails, then generate without OTFB in the tracker and redefine the tracker after with OTFB.
    matched_from_distribution_density_multibunch(beam, SPS_ring, SPS_tracker, distribution_options_list,
                                                 N_bunches, np.around(bunch_positions[:N_bunches]),
                                                 intensity_list=bunch_intensities[:N_bunches],
                                                 n_iterations=6, TotalInducedVoltage=total_imp)
    beam.dt += 1000 * rfstation.t_rf[0,0]

    np.save(lxdir + f'data_files/with_impedance/generated_beams/generated_beam_{fit_type}_{N_bunches}_dE_r.npy', beam.dE)
    np.save(lxdir + f'data_files/with_impedance/generated_beams/generated_beam_{fit_type}_{N_bunches}_dt_r.npy', beam.dt)
else:
    beam.dE = np.load(lxdir + f'data_files/with_impedance/generated_beams/generated_beam_{fit_type}_{N_bunches}_dE_r.npy')
    beam.dt = np.load(lxdir + f'data_files/with_impedance/generated_beams/generated_beam_{fit_type}_{N_bunches}_dt_r.npy') - dphase / rfstation.omega_rf[0,0]


SPS_rf_tracker = RingAndRFTracker(rfstation, beam, TotalInducedVoltage=total_imp,
                                  CavityFeedback=OTFB, Profile=profile)
SPS_tracker = FullRingAndRF([SPS_rf_tracker])

profile.track()
total_imp.induced_voltage_sum()

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

if not GENERATE:
    # Tracking ------------------------------------------------------------------------------------------------------------
    # Tracking with the beam

    if STDY_OSC:
        max_pow = np.zeros(N_t)
        min_pow = np.zeros(N_t)

    # Parameters to save turn-by-turn
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

        if i % dt_track == 0:
            print(i, "intensity:", beam.intensity)

        if STDY_OSC:
            OTFB.OTFB_1.calc_power()
            OTFB.OTFB_2.calc_power()

            max_pow[i], min_pow[i] = dut.oscillation_study(OTFB.OTFB_1.P_GEN[-h:])

        if i % dt_ptrack == 0:
            # Power
            OTFB.OTFB_1.calc_power()
            OTFB.OTFB_2.calc_power()
            max_pow_arr[0, n] = np.max(OTFB.OTFB_1.P_GEN[-h:])
            max_pow_arr[1, n] = np.max(OTFB.OTFB_2.P_GEN[-h:])

            # Antenna voltage
            max_V_arr[0, n] = np.max(np.abs(OTFB.OTFB_1.V_ANT[-h:]))
            max_V_arr[1, n] = np.max(np.abs(OTFB.OTFB_2.V_ANT[-h:]))

            # Beam params
            fwhm_arr[:, n], pos_arr[:, n], pos_fit_arr[:, n], x_72, y_72 = dut.bunch_params(profile,
                                                                                            get_72=False)
            int_arr[n] = beam.intensity
            n += 1

        if i % dt_save == 0:
            dut.plot_params(fwhm_arr, pos_arr, pos_fit_arr,
                            max_pow_arr, max_V_arr, lxdir + sim_dir,
                            rfstation.t_rf[0,0], i, n - 1,
                            MB = not SINGLE_BATCH)

            dut.save_params(fwhm_arr, pos_arr, pos_fit_arr,
                            max_pow_arr, max_V_arr, lxdir + sim_dir)

            dut.plot_ramp(int_arr, i, n - 1, lxdir + sim_dir)

        if i % dt_plot == 0:
            OTFB.OTFB_1.calc_power()
            OTFB.OTFB_2.calc_power()

            if SAVE_RESULTS:
                dut.save_plots_OTFB(OTFB, lxdir + sim_dir + f'fig/', i)
                dut.save_data(OTFB, lxdir + sim_dir + f'sim_data/', i)
                dut.save_profile(profile, lxdir + sim_dir + f'sim_data/', i)

    print("Final intensity:", beam.intensity)
    OTFB.OTFB_1.calc_power()
    OTFB.OTFB_2.calc_power()
    dut.save_plots_OTFB(OTFB, lxdir + sim_dir + f'fig/', N_t)
    dut.plot_ramp(int_arr, N_t, n - 1, lxdir + sim_dir)

    dut.plot_params(fwhm_arr, pos_arr, pos_fit_arr,
                    max_pow_arr, max_V_arr, lxdir + sim_dir,
                    rfstation.t_rf[0, 0], N_t, n - 1,
                    MB = not SINGLE_BATCH)

    dut.save_params(fwhm_arr, pos_arr, pos_fit_arr,
                    max_pow_arr, max_V_arr, lxdir + sim_dir)

    if SAVE_RESULTS:
        dut.save_data(OTFB, lxdir + sim_dir + f'sim_data/', i)

        if not os.path.exists(lxdir + sim_dir + f'generated_beams/'):
            os.makedirs(lxdir + sim_dir + f'generated_beams/')
        np.save(lxdir + sim_dir + f'generated_beams/generated_beam_{fit_type}_{N_bunches}_dE_end_{N_t}.npy', beam.dE)
        np.save(lxdir + sim_dir + f'generated_beams/generated_beam_{fit_type}_{N_bunches}_dt_end_{N_t}.npy', beam.dt)

        if not os.path.exists(lxdir + sim_dir + f'profile_data/'):
            os.makedirs(lxdir + sim_dir + f'profile_data/')
        np.save(lxdir + sim_dir + f'profile_data/generated_profile_{fit_type}_{N_bunches}_end_{N_t}', profile.n_macroparticles)
        np.save(lxdir + sim_dir + f'profile_data/generated_profile_bins_{fit_type}_{N_bunches}_end_{N_t}', profile.bin_centers)

        if STDY_OSC:
            dut.save_osc(max_pow, min_pow, lxdir + sim_dir)

        np.save(lxdir + sim_dir + f"sim_data/induced_voltage", SPS_rf_tracker.totalInducedVoltage.induced_voltage)
        np.save(lxdir + sim_dir + f"sim_data/induced_voltage_time", SPS_rf_tracker.totalInducedVoltage.time_array)

