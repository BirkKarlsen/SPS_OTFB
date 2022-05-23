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
parser.add_argument("--corr_sign", "-cs", type=int,
                    help="Sign of the corrections in phase from the OTFB, default is +1 (+1).")



args = parser.parse_args()


# Imports -------------------------------------------------------------------------------------------------------------

print('Importing...\n')
import matplotlib.pyplot as plt
import numpy as np
import utility_files.data_utilities as dut
import os.path
from datetime import date
import utility_files.analysis_tools as at
from scipy.constants import c
from matplotlib import rc

from blond.llrf.cavity_feedback import SPSCavityFeedback, CavityFeedbackCommissioning
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions_multibunch import matched_from_distribution_density_multibunch
from blond.trackers.tracker import FullRingAndRF, RingAndRFTracker
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage, InducedVoltageTime
from blond.impedances.impedance_sources import InputTable, TravelingWaveCavity
from blond.utils import bmath as bm
from blond.llrf.signal_processing import cartesian_to_polar

# SPS Impedance
from SPS.impedance_scenario import scenario, impedance2blond

# TODO: change back - DONE
fit_type = 'fwhm'
SINGLE_BATCH = True
GENERATE = False                           # TODO: True
SAVE_RESULTS = False
LXPLUS = False                              # TODO: change back before copy to lxplus
SPS_IMP = False
STDY_OSC = False
TRACK_IMP = True
OMEGA_SCENARIO = 3

if not LXPLUS:
    plt.rcParams.update({
        #'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 16
    })
    rc('text', usetex=True)

# Parameters ----------------------------------------------------------------------------------------------------------
C = 2 * np.pi * 1100.009                        # Ring circumference [m]
gamma_t = 18.0                                  # Transition Gamma [-]
alpha = 1 / (gamma_t**2)                        # Momentum compaction factor [-]
p_s = 440e9                                     # Synchronous momentum [eV]
h = 4620                                        # 200 MHz harmonic number [-]
#V = (0.911535 * 4 + 1.526871 * 2) * 1e6         # 200 MHz RF voltage [V]
V = 5.96e6
phi = 0                                         # 200 MHz phase [-]


# Parameters for the Simulation
N_m = int(5e5)                                  # Number of macro-particles for tracking
N_t = 1000                                      # Number of turns to track

dt_plot = 2 # TODO: 1000
dt_track = 10 # TODO: 1000
dt_save = 10 # TODO: 1000
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
    mstdir = "rev_sign/" # TODO: change back to empty string

if args.alpha_comb is not None:
    a_comb = args.alpha_comb
else:
    #a_comb = 63/64
    a_comb = 31/32

if args.llrf_gain is not None:
    llrf_g = args.llrf_gain
else:
    llrf_g = 16 # TODO: 10

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

if args.corr_sign is not None:
    phase_sign = args.corr_sign
else:
    phase_sign = +1

if SINGLE_BATCH:
    N_bunches = 72                              # Number of bunches
else:
    N_bunches = 288                             # Number of bunches


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
    lxdir = "/afs/cern.ch/work/b/bkarlsen/Simulation_Files/BLonD_OTFB_development/"
else:
    lxdir = "../"

if OMEGA_SCENARIO == 1:
    domega = [0, 0.1055e6]
elif OMEGA_SCENARIO == 2:
    domega = [0, 0.2275e6]
else:
    domega = [0, 0]

#domega = [0.18433333e6, 0.2275e6]
domega = [0, 0]
#G_tx = [0.251402590786449, 0.511242728131293]
#G_tx = [0.25154340605790062590, 0.510893981556323]
#G_tx = [0.22909261332041,
#        0.429420301179296]
#G_tx = [0.163212561182363 * 0.8,
#        0.127838041632473 * 0.8]
G_tx = [0.3/0.33,
        0.3/0.33]
G_llrf = 20
#G_tx = [1.0352156647332156,
#        1.077709051028262]
#domega = [0.18433333e6,  # Both at 200.222
#          0.2275e6]

# Objects -------------------------------------------------------------------------------------------------------------
print('Initializing Objects...\n')


# Ring
SPS_ring = Ring(C, alpha, p_s, Proton(), N_t)

# RFStation
SINGLE_RF = True
if SINGLE_RF:
    rfstation = RFStation(SPS_ring, [h], [V], [0], n_rf=1)
else:
    rfstation = RFStation(SPS_ring, [h, 4 * h], [V, 0.19 * V], [0, np.pi], n_rf=2)

print('RF frequency',rfstation.omega_rf[0,0] / 2 / np.pi)

# SINGLE BUNCH FIRST
# Beam
bunch_intensities = np.load(lxdir + 'data_files/beam_parameters/avg_bunch_intensities_red.npy')
total_intensity = 3385.8196 * 10**10

bunch_intensities = total_intensity * bunch_intensities / np.sum(bunch_intensities)  # normalize to 3385.8196 * 10**10
n_macro = N_m * N_bunches * bunch_intensities / np.sum(bunch_intensities)

beam = Beam(SPS_ring, int(np.sum(n_macro[:N_bunches])), int(total_intensity))

# Profile
delta_slice = 0.0
profile = Profile(beam, CutOptions = CutOptions(cut_left=rfstation.t_rf[0,0] * (1000 - 2.5 + delta_slice),
    cut_right=rfstation.t_rf[0,0] * (1000 + 72 * 5 * 4 + 250 * 3 + 125), #* 4 + 250 * 3
    n_slices=int(round(2**7 * (72 * 5 * 4 + 250 * 3 + 125)))))
#profile = Profile(beam, CutOptions = CutOptions(cut_left=0.e-9,
#    cut_right=rfstation.t_rev[0], n_slices=2**7 * 4620))
print(profile.bin_centers[0] / rfstation.t_rf[0,0], profile.bin_size / rfstation.t_rf[0,0])
# One Turn Feedback
V_part = 0.5442095845867135
# TODO: Run with Gtx of 1

#G_tx_ls = [0.2712028956, 0.58279606]
#G_llrf_ls = [41.751786, 35.24865]
#llrf_g = G_llrf_ls

Commissioning = CavityFeedbackCommissioning(open_FF=True, debug=False,
                                            rot_IQ=-1, FIR_filter=1, open_FB=False)
OTFB = SPSCavityFeedback(rfstation, beam, profile, post_LS2=True, V_part=V_part,
                         Commissioning=Commissioning, G_tx=G_tx, a_comb=a_comb,
                         G_llrf=G_llrf, df=domega, G_ff=1)   # TODO: change back to only 20


# Impedance of the SPS
if SPS_IMP:
    freqRes = 43.3e3          # Hz

    modelStr = "futurePostLS2_SPS_noMain200TWC.txt" # TODO: change back to futurePostLS2_SPS_noMain200TWC.txt
    # "200TWC_only.txt"
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
    # Cavities
    l_cav = 32 * 0.374
    v_g = 0.0946
    tau = l_cav / (v_g * c) * (1 + v_g)
    f_cav = 200.222e6
    n_cav = 4  # factor 2 because of two four/five-sections cavities
    short_cavity = TravelingWaveCavity(l_cav ** 2 * n_cav * 27.1e3 / 8,
                                       f_cav, 2 * np.pi * tau)
    shortInducedVoltage = InducedVoltageTime(beam, profile,
                                             [short_cavity])
    l_cav = 43 * 0.374
    tau = l_cav / (v_g * c) * (1 + v_g)
    n_cav = 2
    long_cavity = TravelingWaveCavity(l_cav ** 2 * n_cav * 27.1e3 / 8,
                                      f_cav, 2 * np.pi * tau)
    longInducedVoltage = InducedVoltageTime(beam, profile,
                                            [long_cavity])
    total_imp = TotalInducedVoltage(
        beam, profile, [shortInducedVoltage, longInducedVoltage])
    total_imp.induced_voltage_sum()

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
                                                 N_bunches, bunch_positions[:N_bunches],
                                                 intensity_list=bunch_intensities[:N_bunches],
                                                 n_iterations=6, TotalInducedVoltage=total_imp)
    beam.dt += 1000 * rfstation.t_rf[0,0]

    np.save(lxdir + f'data_files/with_impedance/generated_beams/generated_beam_{fit_type}_{N_bunches}_dE_r.npy', beam.dE)
    np.save(lxdir + f'data_files/with_impedance/generated_beams/generated_beam_{fit_type}_{N_bunches}_dt_r.npy', beam.dt)
else:
    beam.dE = np.load(lxdir + f'data_files/with_impedance/generated_beams/generated_beam_{fit_type}_{N_bunches}_dE_r.npy')
    beam.dt = np.load(lxdir + f'data_files/with_impedance/generated_beams/generated_beam_{fit_type}_{N_bunches}_dt_r.npy') + 0.0 * rfstation.t_rf[0,0]

SPS_rf_tracker = RingAndRFTracker(rfstation, beam, TotalInducedVoltage=total_imp,
                                  CavityFeedback=OTFB, Profile=profile, interpolation=True)
SPS_tracker = FullRingAndRF([SPS_rf_tracker])




SPS_rf_tracker_with_OTFB = RingAndRFTracker(rfstation, beam, TotalInducedVoltage=None,
                                  CavityFeedback=OTFB, Profile=profile, interpolation=True)

SPS_rf_tracker_with_imp = RingAndRFTracker(rfstation, beam, TotalInducedVoltage=total_imp,
                                  CavityFeedback=None, Profile=profile, interpolation=True)
SPS_tracker_w_imp = FullRingAndRF([SPS_rf_tracker_with_imp])




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

print('omega_c', OTFB.OTFB_1.omega_c)
print('omega_r', OTFB.OTFB_1.omega_r)
print('t_rf', rfstation.t_rf[0,0])
print('tau', OTFB.OTFB_1.TWC.tau)
print('T_s', OTFB.OTFB_1.T_s)



if not GENERATE:
    # Tracking ------------------------------------------------------------------------------------------------------------
    # Tracking with the beam
    nn = 100
    dt_p = 10
    for i in range(nn):
        OTFB.track()
        #SPS_tracker.track()
        SPS_rf_tracker_with_OTFB.track()
        #SPS_tracker_w_imp.track()
        profile.track()
        total_imp.induced_voltage_sum()
        if 0 == (i + 1) % dt_p:
            print(i + 1)
        if (i + 1) == nn:
            # Compare generator induced and beam induced contributions
            gen_ind = OTFB.V_sum - OTFB.OTFB_1.V_IND_FINE_BEAM[-profile.n_slices:] \
                      - OTFB.OTFB_2.V_IND_FINE_BEAM[-profile.n_slices:]

            beam_ind = OTFB.OTFB_1.V_IND_FINE_BEAM[-profile.n_slices:] + OTFB.OTFB_2.V_IND_FINE_BEAM[-profile.n_slices:]

            gV, gp = cartesian_to_polar(gen_ind)
            bV, bp = cartesian_to_polar(beam_ind)

            gE = gV * np.sin(rfstation.omega_rf[0, 0] * profile.bin_centers + gp + np.pi/2)
            bE = bV * np.sin(rfstation.omega_rf[0, 0] * profile.bin_centers + bp + np.pi/2)
            print('Calc')


    SPS_rf_tracker.rf_voltage_calculation()
    SPS_rf_tracker_with_OTFB.rf_voltage_calculation()
    SPS_rf_tracker_with_imp.rf_voltage_calculation()
    #SPS_tracker_w_imp.track()

    OTFB_tot = SPS_rf_tracker.rf_voltage - SPS_rf_tracker_with_imp.rf_voltage
    IMP_tot = SPS_rf_tracker_with_imp.totalInducedVoltage.induced_voltage

    NEW_PLOTS = False
    PLOT_MATRIX_ELEMENTS = False
    if NEW_PLOTS:
        plt.figure()
        plt.title('Total Voltage')
        plt.plot(profile.bin_centers, SPS_rf_tracker_with_OTFB.total_voltage, label='OTFB')
        plt.plot(profile.bin_centers, SPS_rf_tracker_with_imp.total_voltage, label='IMP')
        plt.plot(profile.bin_centers,
                 288 * 4e6 * profile.n_macroparticles / np.sum(profile.n_macroparticles),
                 label='profile')
        plt.xlim((4.985e-6, 5.04e-6))
        plt.legend()

        plt.figure()
        plt.title('Beam Induced Voltage')
        plt.plot(profile.bin_centers,
                 SPS_rf_tracker_with_OTFB.rf_voltage - SPS_rf_tracker_with_imp.rf_voltage,
                 label='OTFB')
        plt.plot(profile.bin_centers, SPS_rf_tracker_with_imp.totalInducedVoltage.induced_voltage, label='IMP')
        plt.plot(profile.bin_centers,
                 288 * 4e6 * profile.n_macroparticles / np.sum(profile.n_macroparticles),
                 label='profile')
        plt.xlim((4.985e-6, 5.04e-6))
        plt.legend()

        at.plot_IQ(OTFB.OTFB_1.V_ANT[-h:],
                   OTFB.OTFB_1.V_IND_COARSE_GEN[-h:],
                   OTFB.OTFB_1.V_IND_COARSE_BEAM[-h:],
                   end=1000 + 5 * 72, wind=4e6)

        t_coarse = np.linspace(0, rfstation.t_rev[0], h)
        plt.figure()
        #plt.plot(t_coarse, OTFB.OTFB_1.I_COARSE_BEAM[-h:].real, color='r')
        #plt.plot(t_coarse, OTFB.OTFB_1.I_COARSE_BEAM[-h:].imag, color='b')
        plt.plot(profile.bin_centers, OTFB.OTFB_1.I_FINE_BEAM[-profile.n_slices:].real, color='r')
        plt.plot(profile.bin_centers, OTFB.OTFB_1.I_FINE_BEAM[-profile.n_slices:].imag, color='b')

        plt.figure()
        plt.plot(profile.bin_centers, OTFB.phi_corr)

    else:
        # Compare wake-fields from impedance and OTFB
        plt.figure()
        plt.plot(profile.bin_centers, bE, label='Turn 0')
        plt.plot(profile.bin_centers, OTFB_tot, label='OTFB')
        plt.plot(profile.bin_centers, IMP_tot, label='IMP')
        plt.plot(profile.bin_centers,
                 288 * 4e6 * profile.n_macroparticles / np.sum(profile.n_macroparticles),
                 label='profile')
        plt.xlim((4.985e-6, 5.04e-6))
        plt.legend()


        plt.figure()
        plt.title('Beam-Induced Voltage')
        plt.plot(profile.bin_centers / 1e-6, bE / 1e6, label='OTFB', marker='x', color='r')
        plt.plot(profile.bin_centers / 1e-6, IMP_tot / 1e6, label='IMP', color='b')
        plt.plot(profile.bin_centers / 1e-6,
                 288 * 4e6 * profile.n_macroparticles / np.sum(profile.n_macroparticles) / 1e6,
                 label='profile', color='black')
        plt.xlim((4.985e-6 / 1e-6, 5.04e-6 / 1e-6))
        #plt.xlim((127500 * profile.bin_size, 130000 * profile.bin_size))
        plt.ylabel(r'Induced Voltage [MV]')
        plt.xlabel(r'$\Delta t$ [$\mu$s]')
        plt.legend()


        plt.figure()
        plt.title('Generator voltage only')
        plt.plot(profile.bin_centers, gE, label='OTFB')
        plt.plot(profile.bin_centers, SPS_rf_tracker_with_imp.rf_voltage, label='IMP')
        plt.plot(profile.bin_centers,
                 288 * 4e6 * profile.n_macroparticles / np.sum(profile.n_macroparticles),
                 label='profile')
        plt.xlim((127500 * profile.bin_size, 130000 * profile.bin_size))
        plt.legend()


        at.plot_IQ(OTFB.OTFB_1.V_ANT[-h:],
                   OTFB.OTFB_1.V_IND_COARSE_GEN[-h:],
                   OTFB.OTFB_1.V_IND_COARSE_BEAM[-h:],
                   end=1000 + 5 * 72, wind=4e6)
        plt.title('Phasor plot 3-section')
        plt.xlabel('In-Phase [V]')
        plt.ylabel('Quadrature [V]')

        at.plot_IQ(OTFB.OTFB_2.V_ANT[-h:],
                   OTFB.OTFB_2.V_IND_COARSE_GEN[-h:],
                   OTFB.OTFB_2.V_IND_COARSE_BEAM[-h:],
                   end=1000 + 5 * 72, wind=4e6)
        plt.title('Phasor plot 4-section')
        plt.xlabel('In-Phase [V]')
        plt.ylabel('Quadrature [V]')

        rf_current = OTFB.OTFB_1.I_COARSE_BEAM[-h:]
        rf_current = np.mean(rf_current[1000:1000 + 5 * 72])


        at.plot_IQ_both_cavities(OTFB, end=1000 + 72 * 5)

        #plt.figure()
        #plt.title('Antenna 3-section')
        #plt.plot(np.abs(OTFB.OTFB_1.V_ANT[-h:]))

        #plt.figure()
        #plt.title('Power 3-section')
        #OTFB.OTFB_1.calc_power()
        #plt.plot(OTFB.OTFB_1.P_GEN[-h:])

        #plt.figure()
        #plt.title('Antenna 4-section')
        #plt.plot(np.abs(OTFB.OTFB_2.V_ANT[-h:]))

        #plt.figure()
        #plt.title('Power 4-section')
        #OTFB.OTFB_2.calc_power()
        #plt.plot(OTFB.OTFB_2.P_GEN[-h:])

        #plt.figure()
        #plt.title('Vgen 3-section')
        #plt.plot(np.abs(OTFB.OTFB_1.V_IND_COARSE_GEN[-h:]))

        #plt.figure()
        #plt.title('Vgen 4-section')
        #plt.plot(np.abs(OTFB.OTFB_2.V_IND_COARSE_GEN[-h:]))

        beam_ind, gen_ind = dut.find_induced_and_generator(OTFB, rfstation, profile, SPS_rf_tracker)
        beam_eff_ind = dut.find_effective_induced(OTFB, rfstation, profile, SPS_rf_tracker)

        plt.figure()
        plt.title('Induced voltage in 200 MHz TWC')
        plt.plot((profile.bin_centers - profile.bin_centers[0]) * 1e9, beam_ind * 1e-6, color='b', label='no OTFB')
        plt.plot((profile.bin_centers - profile.bin_centers[0]) * 1e9, beam_eff_ind* 1e-6, color='r', label='with OTFB')
        plt.legend()
        plt.xlim((0, 2500))
        plt.xlabel('$\Delta t$ [ns]')
        plt.ylabel('$V_{ind}$ [MV]')

        plt.figure()
        RF_current = OTFB.OTFB_1.I_FINE_BEAM[-profile.n_slices:]
        beam_ind_volt = OTFB.OTFB_1.V_IND_FINE_BEAM[-profile.n_slices:] + OTFB.OTFB_2.V_IND_FINE_BEAM[-profile.n_slices:]
        plt.plot(profile.bin_centers, profile.n_macroparticles * beam.ratio * 1.6e-19)
        plt.plot(profile.bin_centers,
                 np.abs(RF_current)*profile.bin_size/2 * np.sin(rfstation.omega_rf[0, 0] * profile.bin_centers
                                             + np.angle(RF_current) + np.pi / 2))
        plt.plot(profile.bin_centers,
                 np.abs(beam_ind_volt) * 18e-16 * np.sin(rfstation.omega_rf[0,0] * profile.bin_centers
                                                + np.angle(beam_ind_volt) + np.pi / 2))
        plt.plot(profile.bin_centers,
                 IMP_tot * 18e-16)

        fwhm_arr, pos_arr, pos_fit_arr, x_72, y_72 = dut.bunch_params(profile, get_72=False)
        dut.plot_bbb_offset(pos_fit_arr, 1, '', 0, show=True)



    if PLOT_MATRIX_ELEMENTS:
        # Cavities
        l_cav = 32 * 0.374
        v_g = 0.0946
        tau = l_cav / (v_g * c) * (1 + v_g)
        f_cav = 200.222e6
        f_carrier = rfstation.omega_rf[0,0] / 2 / np.pi
        domega = 2 * np.pi * (f_carrier - f_cav)

        t_coarse = np.linspace(0, rfstation.t_rev[0], h)
        t_fine = np.linspace(0, profile.n_slices * profile.bin_size, profile.n_slices)

        hgs, hgc = at.generator_matrix(t_coarse, domega, tau)
        hbs, hbc = at.beam_matrix(t_fine, domega, tau)

        # Generator:
        plt.figure('Generator Matrix')
        plt.title('Generator Matrix')
        plt.plot(t_coarse / tau, OTFB.OTFB_1.TWC.h_gen.real, label=r'$h_{gs}$')
        plt.plot(t_coarse / tau, -OTFB.OTFB_1.TWC.h_gen.imag, label=r'$-h_{gc}$')
        plt.plot(t_coarse / tau, hgs * 1e4, label=r'$h_{gs}$ p')
        plt.plot(t_coarse / tau, -hgc * 1e4, label=r'$-h_{gc}$ p')
        plt.legend()

        plt.figure('Beam Matrix')
        plt.title('Beam Matrix')
        plt.plot(t_fine / tau, OTFB.OTFB_1.TWC.h_beam.real, label=r'$h_{bs}$')
        plt.plot(t_fine / tau, -OTFB.OTFB_1.TWC.h_beam.imag, label=r'$-h_{bc}$')
        plt.plot(t_fine / tau, hbs * 1e6 / 2, label=r'$h_{bs}$')
        plt.plot(t_fine / tau, -hbc * 1e6 / 2, label=r'$-h_{bc}$')
        plt.legend()


    plt.show()


    #voltages = np.ascontiguousarray(SPS_rf_tracker.voltage[:, SPS_rf_tracker.counter[0]])
    #omega_rf = np.ascontiguousarray(SPS_rf_tracker.omega_rf[:, SPS_rf_tracker.counter[0]])
    #phi_rf = np.ascontiguousarray(SPS_rf_tracker.phi_rf[:, SPS_rf_tracker.counter[0]])

    #un_comp_rf_voltage = bm.rf_volt_comp(voltages, omega_rf, phi_rf,
    #                                          SPS_rf_tracker.profile.bin_centers)

    #plt.figure()
    #plt.plot(SPS_rf_tracker.rf_voltage, label='OTFB induced')
    #plt.plot(SPS_rf_tracker.total_voltage, label='total voltage')
    #plt.plot(un_comp_rf_voltage, label='uncomp rf volt')
    #plt.xlim((140500, 142000))
    #plt.legend()


    #plt.figure()
    #plt.plot(profile.bin_centers, SPS_rf_tracker.totalInducedVoltage.induced_voltage, label='from impedance model')
    #plt.plot(profile.bin_centers, (SPS_rf_tracker.rf_voltage - un_comp_rf_voltage), label='from OTFB')
    #plt.plot(profile.bin_centers,
    #         288 * 4e6 * profile.n_macroparticles / np.sum(profile.n_macroparticles),
    #         label='profile')
    #plt.xlim((127500 * profile.bin_size, 130000 * profile.bin_size))
    #plt.legend()

    #plt.figure()
    #plt.plot(SPS_rf_tracker.cavityFB.V_corr, label='Tracker')
    #plt.plot(OTFB.V_corr, label='OTFB')

    #plt.figure()
    #plt.plot(SPS_rf_tracker.cavityFB.phi_corr * 180 / np.pi, label='Tracker')
    #plt.plot(OTFB.phi_corr * 180 / np.pi, label='OTFB')
    #plt.legend()

    #at.plot_IQ(OTFB.OTFB_1.V_ANT[-OTFB.OTFB_1.n_coarse:],
    #           OTFB.OTFB_1.V_IND_COARSE_GEN[-OTFB.OTFB_1.n_coarse:],
    #           OTFB.OTFB_1.V_IND_COARSE_BEAM[-OTFB.OTFB_1.n_coarse:],
    #           wind=4e6)

    plt.show()

    n = 0
    for i in range(0):
        SPS_tracker.track()
        profile.track()
        total_imp.induced_voltage_sum()
        OTFB.track()

        if i % dt_track == 0:
            print(i, "intensity:", beam.intensity)

        if STDY_OSC:
            OTFB.OTFB_1.calc_power()
            OTFB.OTFB_2.calc_power()


        if i % dt_ptrack == 0:
            # Power
            OTFB.OTFB_1.calc_power()
            OTFB.OTFB_2.calc_power()

            n += 1


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

        np.save(lxdir + sim_dir + f"sim_data/induced_voltage", SPS_rf_tracker.totalInducedVoltage.induced_voltage)
        np.save(lxdir + sim_dir + f"sim_data/induced_voltage_time", SPS_rf_tracker.totalInducedVoltage.time_array)

