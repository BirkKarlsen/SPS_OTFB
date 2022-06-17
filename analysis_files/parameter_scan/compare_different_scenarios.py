'''
File to compare the power and bunch-by-bunch offset across different scenarios

Author: Birk Emil Karlsen-Bæck
'''

# Import ---------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
from analysis_files.measurement_analysis.sps_cable_transferfunction_script import cables_tranfer_function

# Utility files
import utility_files.data_utilities as dut
import utility_files.analysis_tools as at


# Functions from related files
from analysis_files.measurement_analysis.import_data import get_power


plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 16
    })


def measured_offset():
    pos_tot = np.load('../../data_files/beam_measurements/bunch_positions_total_red.npy')
    pos_fl = pos_tot.reshape((25 * 100, 288))
    pos_fb = pos_fl[:,:72]
    b_n = np.linspace(1, 72, 72)
    pds = np.zeros(pos_fb.shape)

    for i in range(pos_fb.shape[0]):
        s1, i1, rval, pval, stderr = linregress(b_n, pos_fb[i,:])

        pds[i,:] = pos_fb[i,:] - s1 * b_n - i1

    avg_pd = np.mean(pds, axis = 0)
    std_pd = np.std(pds, axis = 0)

    return avg_pd, std_pd




beam_parameter = 'Bunch Position'

files = ['_3000_fr1_ve88.npy', '_3000_fr3_ve88.npy', '_10000_fr2_ve88.npy',
         '_12000_fr1_ve101.npy', '_16000_fr3_ve101.npy', '_29000_fr2_ve101.npy']

sim_name = [r'2021 $f_r$, $V_a = 5.89$ MV', r'2018 $f_r$, $V_a = 5.89$ MV', r'design $f_r$, $V_a = 5.89$ MV',
            r'2021 $f_r$, $V_a = 6.76$ MV', r'2018 $f_r$, $V_a = 6.76$ MV', r'design $f_r$, $V_a = 6.76$ MV']

bunches = np.array([1])
batch_length = 72
number_of_batches = 4
until_turn = 30000#29000
T_rev = 4620 / 200.394e6
distance = 20
tb = 4.990159369074305e-09

normal_buckets = np.linspace(0, 71 * 5, 72)
normal_buckets = np.concatenate((normal_buckets, np.linspace(0, 71 * 5, 72) + 50 + normal_buckets[-1]))
normal_buckets = np.concatenate((normal_buckets, np.linspace(0, 71 * 5, 72) + 50 + normal_buckets[-1]))
normal_buckets = np.concatenate((normal_buckets, np.linspace(0, 71 * 5, 72) + 50 + normal_buckets[-1]))
normal_buckets = normal_buckets + 1000

normal_buckets *= 4.990159369074305e-09


choose_batch = 0

PLT_BP = True
PLT_BL = False
COMPARE_DIFFERENT = False
COMPARE_CTF = True
APPLY_CTF = True
PRT_PP_DIFF = True


# Find files -----------------------------------------------------------------------------------------------------------
dir_current_file = os.path.dirname(os.path.abspath(__file__))
data_files_dir = dir_current_file[:-len('analysis_files/parameter_scan')] + \
                 'data_files/beam_parameters_tbt/200MHz_volt_scan_power_and_profile/'

# Power
sample_data = np.load(data_files_dir + '3sec_power' + files[0])

power3 = np.zeros((sample_data.shape[0], len(files)))
power4 = np.zeros((sample_data.shape[0], len(files)))

for i in range(len(files)):
    power3[:, i] = np.load(data_files_dir + '3sec_power' + files[i])
    power4[:, i] = np.load(data_files_dir + '4sec_power' + files[i])


# profile
sample_data = np.load(data_files_dir + f'profile' + files[0])

profiles = np.zeros((sample_data.shape[0], len(files)))
bins = np.zeros((sample_data.shape[0], len(files)))

for i in range(len(files)):
    sample_i = np.load(data_files_dir + 'profile' + files[i])
    profiles[:, i] = sample_i[:, 0]
    bins[:, i] = sample_i[:, 1]

profiles_wo_CTF = np.copy(profiles)

if APPLY_CTF:
    for i in range(bins.shape[1]):
        CTF_profile_i = cables_tranfer_function(bins[:,i], profiles[:,i])
        profiles[:,i] = CTF_profile_i * np.max(profiles[:,i]) / np.max(CTF_profile_i)


print('Finding bunch positions...\n')
N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit, \
    Bunch_peaksFit, Bunch_Exponent, Goodness_of_fit, x_71, y_71 \
        = dut.getBeamPattern_3(bins[:,0], profiles,
                           distance=2**7 * 3, fit_option='fwhm', heightFactor=50,
                           wind_len=5, save_72_fits=False)

N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit_wo_CTF, \
    Bunch_peaksFit, Bunch_Exponent, Goodness_of_fit, x_71, y_71 \
        = dut.getBeamPattern_3(bins[:,0], profiles_wo_CTF,
                           distance=2**7 * 3, fit_option='fwhm', heightFactor=50,
                           wind_len=5, save_72_fits=False)


bbb_offsets = np.zeros(Bunch_positionsFit[:,:72].T.shape)
bbb_offsets_wo_CTF = np.zeros(Bunch_positionsFit_wo_CTF[:,:72].T.shape)
xs = np.zeros(Bunch_positionsFit[:,:72].T.shape)


print('Computing bunch-by-bunch offset...\n')
for i in range(len(files)):
    bbb_offsets[:, i] = at.find_offset(Bunch_positionsFit[i,:72])
    bbb_offsets_wo_CTF[:, i] = at.find_offset(Bunch_positionsFit_wo_CTF[i,:72])
    xs[:, i] = np.linspace(0, len(Bunch_positionsFit[0,:72]), len(Bunch_positionsFit[0,:72]))


bbb_offsets = bbb_offsets
xs = xs




# Make the plot ---------------------------------------------------------------
CAV_TYPE = 3
# Power arrays
Ns = len(files)
dt = 8e-9
t = np.linspace(0, dt * 65536, 65536)
dts = ((5.981e-6 - 4.983e-6) + (13.643e-6 - 12.829e-6)) / 2
dts1 = ((5.975 - 5.888) + (7.542 - 7.443)) / 2 * 1e-6
ts = np.linspace(0, 4.990159369074305e-09 * 4620, 4620) + dts + dts1
t_s = 1e6
P_s = 1e-3
sec3_mean_tot, sec3_std_tot, sec4_mean_tot, sec4_std_tot = get_power()

# Bunch-by-bunch offset arrays
m, ms = measured_offset()

plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 9
    })
lw = 1


# Making the actual plot
fig, ax = plt.subplots(3, 1, figsize=(5, 7.5))


cmap = plt.get_cmap('jet')
colors = cmap(np.linspace(0, 1, Ns))
#fig.suptitle(f'$f_r =$ 200.1 MHz, $V =$ 5.96 MV')

# Power plot
ax[1].set_title('Power, 3-section')
ax[1].set_ylabel(r'Power [kW]')
ax[1].set_xlabel(r'$\Delta t$ [$\mu$s]')

ax[1].plot(t * t_s, sec3_mean_tot * P_s, color='black', linestyle='--', label='M', linewidth=lw)
# plt.plot(t * t_s, sec3_mean_tot * P_s - shift_P * sec3_mean_tot * P_s, color='b', linestyle='--', label='M')
ax[1].fill_between(t * t_s, (sec3_mean_tot * 0.80) * P_s, (sec3_mean_tot * 1.20) * P_s, alpha=0.3, color='black')

for i in range(len(files)):
    ax[1].plot(ts * t_s, power3[:, i] * P_s, label=f'{sim_name[i]}', color=colors[i], linewidth=lw)

ax[1].set_xlim((3.25e-6 * t_s, 7.8e-6 * t_s))

# Power plot
ax[2].set_title('Power, 4-section')
ax[2].set_ylabel(r'Power [kW]')
ax[2].set_xlabel(r'$\Delta t$ [$\mu$s]')


ax[2].plot(t * t_s, sec4_mean_tot * P_s, color='black', linestyle='--', label='M', linewidth=lw)
# plt.plot(t * t_s, sec4_mean_tot * P_s - shift_P * sec4_mean_tot * P_s, color='b', linestyle='--', label='M')
ax[2].fill_between(t * t_s, (sec4_mean_tot * 0.80) * P_s, (sec4_mean_tot * 1.20) * P_s, alpha=0.3, color='black')

for i in range(len(files)):
    ax[2].plot(ts * t_s, power4[:, i] * P_s, label=f'{sim_name[i]}', color=colors[i], linewidth=lw)

ax[2].set_xlim((3.25e-6 * t_s, 7.8e-6 * t_s))

#

# Bunch-by-bunch offset plot
ax[0].set_title('Bunch-by-bunch')
ax[0].set_ylabel(r'$\Delta t$ [ps]')
ax[0].set_xlabel(r'Bunch Number [-]')

cmap = plt.get_cmap('jet')
colors = cmap(np.linspace(0, 1, Ns))

ax[0].fill_between(xs[:,0], (m - ms) * 1e3, (m + ms) * 1e3, color='black', alpha=0.3)
ax[0].plot(xs[:,0], m * 1e3, linestyle='--', color='black', alpha=1, label='M', linewidth=lw)

for i in range(len(files)):
    ax[0].plot(xs[:, i], bbb_offsets[:, i] * 1e3, label=f'{sim_name[i]}', color=colors[i], linewidth=lw)



handles, labels = ax[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.055), ncol=2)


# Bunch-by-bunch offset plot
JUST_BBB = False
if JUST_BBB:
    plt.figure()
    plt.title('Bunch-by-bunch Offset from Measurements')
    plt.ylabel(r'$\Delta t$ [ps]')
    plt.xlabel(r'Bunch Number [-]')

    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1, Ns))

    plt.fill_between(xs[:,0], (m - ms) * 1e3, (m + ms) * 1e3, color='black', alpha=0.3)
    plt.plot(xs[:,0], m * 1e3, linestyle='--', color='black', alpha=1, label='M')


if COMPARE_CTF:
    choose_sim = 5
    CHOOSE = True

    plt.figure()
    plt.title('Bunch-by-bunch Offset with/without Cable Transfer Function')
    plt.ylabel(r'$\Delta t$ [ps]')
    plt.xlabel(r'Bunch Number [-]')

    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1, Ns))

    plt.fill_between(xs[:, 0], (m - ms) * 1e3, (m + ms) * 1e3, color='black', alpha=0.3)
    plt.plot(xs[:, 0], m * 1e3, linestyle='--', color='black', alpha=1, label='M')

    if CHOOSE:
        plt.plot(xs[:, choose_sim], bbb_offsets[:, choose_sim] * 1e3,
                 label=f'{sim_name[choose_sim]}', color=colors[choose_sim])
        plt.plot(xs[:, choose_sim], bbb_offsets_wo_CTF[:, choose_sim] * 1e3,
                 color=colors[choose_sim], linestyle='--')
    else:
        for i in range(len(files)):
            plt.plot(xs[:, i], bbb_offsets[:, i] * 1e3, label=f'{sim_name[i]}', color=colors[i])
            plt.plot(xs[:, i], bbb_offsets_wo_CTF[:, i] * 1e3, color=colors[i], linestyle='--')

    #plt.legend()


if PRT_PP_DIFF:
    sec3_pp_meas = np.max(sec3_mean_tot) - np.mean(sec3_mean_tot[:100])
    sec3_nob = np.mean(sec3_mean_tot[:100])
    sec4_pp_meas = np.max(sec4_mean_tot) - np.mean(sec4_mean_tot[:100])
    sec4_nob = np.mean(sec4_mean_tot[:100])
    pwr3_pp = np.max(power3, axis=0) - np.mean(power3[:100, :], axis=0)
    pwr3_nob = np.mean(power3[:100, :], axis=0)
    pwr4_pp = np.max(power4, axis=0) - np.mean(power4[:100, :], axis=0)
    pwr4_nob = np.mean(power4[:100, :], axis=0)

    print('Peak-to-peak power difference')
    print('3-section')
    print(f'\tMeasurement: {sec3_pp_meas * P_s:.2f} kW, {100 * sec3_pp_meas / sec3_nob:.2f} %')
    for i in range(len(pwr3_pp)):
        print(f'\t{sim_name[i]}: {pwr3_pp[i] * P_s:.2f} kW, {100 * pwr3_pp[i] / pwr3_nob[i]:.2f} %')

    print('4-section')
    print(f'\tMeasurement: {sec4_pp_meas * P_s:.2f} kW, {100 * sec4_pp_meas / sec4_nob:.2f} %')
    for i in range(len(pwr4_pp)):
        print(f'\t{sim_name[i]}: {pwr4_pp[i] * P_s:.2f} kW, {100 * pwr4_pp[i] / pwr4_nob[i]:.2f} %')





plt.show()