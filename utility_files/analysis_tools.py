'''
Tools to analyse data

author: Birk Emil Karlsen-Bæck
'''
import matplotlib.pyplot as plt
import numpy as np
from blond_common.fitting.profile import binomial_amplitudeN_fit, FitOptions
from blond_common.interfaces.beam.analytic_distribution import binomialAmplitudeN
from blond.llrf.signal_processing import polar_to_cartesian
from scipy.signal import find_peaks, fftconvolve
from scipy.stats import linregress
from scipy.interpolate import interp1d
import json


def fwhm(x, y, level=0.5):
    offset_level = np.mean(y[0:5])
    amp = np.max(y) - offset_level
    t1, t2 = interp_f(x, y, level)
    mu = (t1 + t2) / 2.0
    sigma = (t2 - t1) / 2.35482
    popt = (mu, sigma, amp)

    return popt


def interp_f(time, bunch, level):
    bunch_th = level * bunch.max()
    time_bet_points = time[1] - time[0]
    taux = np.where(bunch >= bunch_th)
    taux1, taux2 = taux[0][0], taux[0][-1]
    t1 = time[taux1] - (bunch[taux1] - bunch_th) / (bunch[taux1] - bunch[taux1 - 1]) * time_bet_points
    t2 = time[taux2] + (bunch[taux2] - bunch_th) / (bunch[taux2] - bunch[taux2 + 1]) * time_bet_points

    return t1, t2


def getBeamPattern_3(timeScale, frames, heightFactor=0.015, distance=500, N_bunch_max=3564,
                     fit_option='fwhm', plot_fit=False, baseline_length = 1, BASE = False,
                     wind_len = 10, save_72_fits = False):
    dt = timeScale[1] - timeScale[0]
    fit_window = int(round(wind_len * 1e-9 / dt / 2))
    N_frames = frames.shape[1]
    N_bunches = np.zeros((N_frames,), dtype=int)
    Bunch_positions = np.zeros((N_frames, N_bunch_max))
    Bunch_lengths = np.zeros((N_frames, N_bunch_max))
    Bunch_peaks = np.zeros((N_frames, N_bunch_max))
    Bunch_intensities = np.zeros((N_frames, N_bunch_max))
    Bunch_positionsFit = np.zeros((N_frames, N_bunch_max))
    Bunch_peaksFit = np.zeros((N_frames, N_bunch_max))
    Bunch_Exponent = np.zeros((N_frames, N_bunch_max))
    Goodness_of_fit = np.zeros((N_frames, N_bunch_max))

    for i in np.arange(N_frames):
        frame = frames[:, i]
        # pos, _ = find_peaks(frame,height=np.max(frames[:,i])*heightFactor,distance=distance)
        pos, _ = find_peaks(frame, height=heightFactor, distance=distance)
        N_bunches[i] = len(pos)
        Bunch_positions[i, 0:N_bunches[i]] = timeScale[pos]
        Bunch_peaks[i, 0:N_bunches[i]] = frame[pos]

        for j, v in enumerate(pos):
            x = 1e9 * timeScale[v - fit_window:v + fit_window]
            y = frame[v - fit_window:v + fit_window]
            if BASE:
                baseline = np.mean(y[:baseline_length])
                y = y - baseline
            try:
                if fit_option == 'fwhm':
                    (mu, sigma, amp) = fwhm(x, y, level=0.5)
                    if j == 71 and save_72_fits:
                        x_71 = x
                        y_71 = y
                    if not save_72_fits:
                        x_71 = x
                        y_71 = y
                #                    (mu2, sigma2, amp2) = fwhm(x,y,level=0.95)
                else:
                    (amp, mu, sigma, exponent) = binomial_amplitudeN_fit(x, y)
                    y_fit = binomialAmplitudeN(x, *[amp, mu, sigma, exponent])


                    if plot_fit: #or exponent > 5:
                        print(amp, mu, sigma, exponent)

                        plt.plot(x, y, label='measurement')
                        plt.plot(x, y_fit, label='fit')
                        plt.vlines(x[baseline_length], np.min(y), np.max(y), linestyle='--')
                        plt.legend()
                        plt.show()

                    sigma /= 4
            except:
                print(len(x), len(y))
                print(x, y)


            Bunch_lengths[i, j] = 4 * sigma
            Bunch_intensities[i, j] = np.sum(y)
            Bunch_positionsFit[i, j] = mu
            Bunch_peaksFit[i, j] = amp
            if fit_option != 'fwhm':
                Bunch_Exponent[i, j] = exponent
                Goodness_of_fit[i, j] = np.mean(np.abs(y - y_fit)/np.max(y)) * 100

    N_bunches_max = np.max(N_bunches)
    Bunch_positions = Bunch_positions[:, 0:N_bunches_max]
    Bunch_peaks = Bunch_peaks[:, 0:N_bunches_max]
    Bunch_lengths = Bunch_lengths[:, 0:N_bunches_max]
    Bunch_intensities = Bunch_intensities[:, 0:N_bunches_max]
    Bunch_positionsFit = Bunch_positionsFit[:, 0:N_bunches_max]
    Bunch_peaksFit = Bunch_peaksFit[:, 0:N_bunches_max]
    Bunch_Exponent = Bunch_Exponent[:, 0:N_bunches_max]
    Goodness_of_fit = Goodness_of_fit[:, 0:N_bunches_max]

    return N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit, \
           Bunch_peaksFit, Bunch_Exponent, Goodness_of_fit, x_71, y_71





def import_measured_profile(fname, turn = 0):
    data = np.load(fname)
    data = np.reshape(data, (100, 99999))

    return data[turn,:]



def positions_measured(prof, bin):
    gen_prof = np.array([prof])
    N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit, \
    Bunch_peaksFit, Bunch_Exponent, Goodness_of_fit, x_71, y_71 \
        = getBeamPattern_3(bin, gen_prof.T,
                           distance=5 * 4, fit_option='fwhm', heightFactor=4e-5,
                           save_72_fits=False, wind_len=5)

    return Bunch_positionsFit[0,:]



def positions_simulated(prof, bin):
    gen_prof = np.array([prof])
    N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit, \
    Bunch_peaksFit, Bunch_Exponent, Goodness_of_fit, x_71, y_71 \
        = getBeamPattern_3(bin, gen_prof.T,
                           distance=2**7 * 4, fit_option='fwhm', heightFactor=4e-5,
                           save_72_fits=False, wind_len=5)

    return Bunch_positionsFit[0,:]


def find_offset_fit(pos1, pos2):
    x = np.linspace(0, len(pos1), len(pos1))

    sl, inter, pval, rval, err = linregress(x, pos1)
    fit_line1 = sl * x + inter

    sl, inter, pval, rval, err = linregress(x, pos2)
    fit_line2 = sl * x + inter

    print(sl)

    offset_fit1 = pos1 - fit_line1
    offset_fit2 = pos2 - fit_line2
    return offset_fit1, offset_fit2


def find_offset_trf(pos1, pos2, t_rf):
    x = np.linspace(0, len(pos1), len(pos1))

    line = 5 * t_rf * x + pos1[0]

    print(5 * t_rf)

    offset1 = pos1 - line
    offset2 = pos2 - line

    return offset1, offset2


def find_offset(pos):
    r'''
    Takes in an array of bunch positions and does a linear regression of them.
    The bunch-by-bunch offset is then calculated by taking the difference between the two.
    :param pos: numpy-array - Ordered list of bunch positions in time
    :return: numpy-array of the bunch-by-bunch offset
    '''
    x = np.linspace(0, len(pos), len(pos))

    sl, inter, pval, rval, err = linregress(x, pos)
    fit_line = sl * x + inter

    offset_fit = pos - fit_line
    return offset_fit


def import_OTFB_signals(data_dir, cfg_dir, turn):
    r'''
    Imports the antenna, generator induced and beam induced voltages from a directory from a specified turn for both
    the 3-section and 4-section cavities.

    :param data_dir: string
        Directory of the data
    :param cfg_dir: string
        Directory of the specific simulation
    :param turn: int
        Turn to import from
    :return:
        Return all the signals that where imported
    '''
    Vant3 = np.load(data_dir + cfg_dir + f'3sec_Vant_{turn}.npy')
    Vgen3 = np.load(data_dir + cfg_dir + f'3sec_Vindgen_{turn}.npy')
    Vbeam3 = np.load(data_dir + cfg_dir + f'3sec_Vindbeam_{turn}.npy')

    Vant4 = np.load(data_dir + cfg_dir + f'4sec_Vant_{turn}.npy')
    Vgen4 = np.load(data_dir + cfg_dir + f'4sec_Vindgen_{turn}.npy')
    Vbeam4 = np.load(data_dir + cfg_dir + f'4sec_Vindbeam_{turn}.npy')

    return Vant3, Vgen3, Vbeam3, Vant4, Vgen4, Vbeam4


def plot_compare_IQs(sigs1, sigs2, titstr='', start=1000, end=3040, norm = False):
    r'''
    Plots the IQ-vectors from two differene simulations in the same plots. Both one with the signals as they are
    and one where the sign of the second set of signals are reversed.

    :param sigs1: 2D numpy-array - Signals from first simulation
    :param sigs2: 2D numpy-array - Signals from second simulation
    :param titstr: string - Title for the plot
    :param start: int - Start index for the averaging of the signals to make the IQ-vectors
    :param end: int - End indec for the averaging of the signals to make the IQ-vectors
    :param norm: bool - Option to make the vectors normalized in the plots
    '''
    if norm:
        for i in range(len(sigs1[0,:])):
            sigs1[:, i] = sigs1[:, i] / np.sum(np.abs(sigs1[:, i]))
            sigs2[:, i] = sigs2[:, i] / np.sum(np.abs(sigs2[:, i]))

    vecs1 = np.zeros((2, sigs1.shape[1]), dtype=complex)
    vecs2 = np.zeros((2, sigs2.shape[1]), dtype=complex)

    for i in range(len(sigs1[0,:])):
        vecs1[:, i] = np.array([0 + 0 * 1j, np.mean(sigs1[start:end, i])], dtype=complex)
        vecs2[:, i] = np.array([0 + 0 * 1j, np.mean(sigs2[start:end, i])], dtype=complex)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(titstr)

    ax[0].plot(vecs1[:,0].real, vecs1[:,0].imag, color='r', label='Vant +1')
    ax[0].plot(vecs1[:,1].real, vecs1[:,1].imag, color='black', label='Vgen +1')
    ax[0].plot(vecs1[:,2].real, vecs1[:,2].imag, color='b', label='Vbeam +1')
    ax[0].plot(vecs2[:,0].real, vecs2[:,0].imag, color='r', label='Vant -1', linestyle='--')
    ax[0].plot(vecs2[:,1].real, vecs2[:,1].imag, color='black', label='Vgen -1', linestyle='--')
    ax[0].plot(vecs2[:,2].real, vecs2[:,2].imag, color='b', label='Vbeam -1', linestyle='--')
    ax[0].legend()

    ax[1].plot(vecs1[:,0].real, vecs1[:,0].imag, color='r', label='Vant +1')
    ax[1].plot(vecs1[:,1].real, vecs1[:,1].imag, color='black', label='Vgen +1')
    ax[1].plot(vecs1[:,2].real, vecs1[:,2].imag, color='b', label='Vbeam +1')
    ax[1].plot(-vecs2[:,0].real, -vecs2[:,0].imag, color='r', label='Vant -1', linestyle='--')
    ax[1].plot(-vecs2[:,1].real, -vecs2[:,1].imag, color='black', label='Vgen -1', linestyle='--')
    ax[1].plot(-vecs2[:,2].real, -vecs2[:,2].imag, color='b', label='Vbeam -1', linestyle='--')
    ax[1].legend()

    #handles, labels = ax.get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper center')



def plot_IQ(Va, Vg, Vb, titstr = '', start=1000, end=3040, norm = False, wind = 3.1e6):
    r'''
    Makes a figure object that plots mean of the antenna, generator induced and beam induced voltages from index start
    to index end. Option to normalize the length of the vectors and change the x- and y-limits of the plot.

    :param Va: numpy-array
        Antenna voltage
    :param Vg: numpy-array
        Generator induced voltage
    :param Vb: numpy-array
        Beam induced voltage
    :param titstr: string
        Title of the plot
    :param start: int
        The first index to include in the mean of the arrays
    :param end: int
        The last index to include in the mean of the arrays
    :param norm: bool
        option to normalize the length of the vectors
    :param wind: float
        the x- and y-limits of the plot
    '''
    if norm:
        Va = Va / np.sum(np.abs(Va))
        Vg = Vg / np.sum(np.abs(Vg))
        Vb = Vb / np.sum(np.abs(Vb))

    Va = np.array([0 + 0 * 1j, np.mean(Va[start:end])], dtype=complex)
    Vg = np.array([0 + 0 * 1j, np.mean(Vg[start:end])], dtype=complex)
    Vb = np.array([0 + 0 * 1j, np.mean(Vb[start:end])], dtype=complex)

    plt.figure()
    plt.title(titstr)
    plt.plot(Va.real, Va.imag, color='r', label='Vant')
    plt.plot(Vg.real, Vg.imag, color='black', label='Vgen')
    plt.plot(Vb.real, Vb.imag, color='b', label='Vbeam')
    plt.xlim((-wind, wind))
    plt.ylim((-wind, wind))
    plt.legend()
    plt.grid()


def plot_IQ_both_cavities(OTFB, start=1000, end=3040, norm=False, xlims=None, ylims=None):
    Va1 = OTFB.OTFB_1.V_ANT[-OTFB.OTFB_1.n_coarse:]
    Vg1 = OTFB.OTFB_1.V_IND_COARSE_GEN[-OTFB.OTFB_1.n_coarse:]
    Vb1 = OTFB.OTFB_1.V_IND_COARSE_BEAM[-OTFB.OTFB_1.n_coarse:]

    Va1 = np.array([0 + 0 * 1j, np.mean(Va1[start:end])], dtype=complex)
    Vg1 = np.array([0 + 0 * 1j, np.mean(Vg1[start:end])], dtype=complex)
    Vb1 = np.array([0 + 0 * 1j, np.mean(Vb1[start:end])], dtype=complex)

    Va2 = OTFB.OTFB_2.V_ANT[-OTFB.OTFB_2.n_coarse:]
    Vg2 = OTFB.OTFB_2.V_IND_COARSE_GEN[-OTFB.OTFB_2.n_coarse:]
    Vb2 = OTFB.OTFB_2.V_IND_COARSE_BEAM[-OTFB.OTFB_2.n_coarse:]

    Va2 = np.array([0 + 0 * 1j, np.mean(Va2[start:end])], dtype=complex)
    Vg2 = np.array([0 + 0 * 1j, np.mean(Vg2[start:end])], dtype=complex)
    Vb2 = np.array([0 + 0 * 1j, np.mean(Vb2[start:end])], dtype=complex)

    fig, ax = plt.subplots(2, 1, figsize=(6.5, 6))

    V_s = 1e-6
    Xf1 = 3.5
    Xf2 = 3

    ax[0].set_title('3-section')
    ax[0].plot(Va1.real * V_s, Va1.imag * V_s, color='r', label=r'$V_{\textrm{ant}}$')
    ax[0].plot(Vg1.real * V_s, Vg1.imag * V_s, color='black', label=r'$V_{\textrm{gen}}$')
    ax[0].plot(Vb1.real * V_s, Vb1.imag * V_s, color='b', label=r'$V_\textrm{beam}$')
    ax[0].set_xlabel('In-phase [MV]')
    ax[0].set_ylabel('Quadrature [MV]')
    ax[0].set_xlim((-Xf1, Xf1))
    ax[0].set_ylim((0, Xf1))
    ax[0].grid()

    ax[1].set_title('4-section')
    ax[1].plot(Va2.real * V_s, Va2.imag * V_s, color='r', label=r'$V_\textrm{ant}$')
    ax[1].plot(Vg2.real * V_s, Vg2.imag * V_s, color='black', label=r'$V_\textrm{gen}$')
    ax[1].plot(Vb2.real * V_s, Vb2.imag * V_s, color='b', label=r'$V_\textrm{beam}$')
    ax[1].set_xlabel('In-phase [MV]')
    ax[1].set_ylabel('Quadrature [MV]')
    ax[1].set_xlim((-Xf2, Xf2))
    ax[1].set_ylim((0, Xf2))
    ax[1].grid()

    fig.suptitle('I/Q-voltages')

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='right')


def import_profiles_from_turn(data_dir, cfg_dir, turn):
    r'''
    Loads the profile for turn turn from the data_dir with configuration cfg_dir and simulation length length_dir.

    :param data_dir: string
        Directory of the data
    :param cfg_dir: string
        Directory of the configuration
    :param turn: int
        Turn from which the profile is recorded from
    :return:
        profile-array and time-array
    '''

    p_profile = np.load(data_dir + cfg_dir + f'profile_{turn}.npy')
    return p_profile[:, 0], p_profile[:, 1]


def plot_profiles(p_profile, p_bin, m_profile, m_bin):
    r'''
    Makes a figure-object that plots two profiles.

    :param p_profile: numpy-array
        Profile array with positive sign
    :param p_bin: numpy-array
        Time array with the positive signed profile
    :param m_profile: numpy-array
        Profile array with negative sign
    :param m_bin: numpy-array
        Time array with the negative signed profile
    '''
    plt.figure()

    plt.plot(p_bin, p_profile, color='r', label='1')
    plt.plot(m_bin, m_profile, color='b', label='-1')
    plt.legend()


def import_and_normalize_profile(dir_str, norm=1):
    r'''
    Imports a profile and its t-axis from a .npy-file where the first column is the profile
    and the second column is the bins. It will also normalize the profile and you have the option to
    scale the resulting normalized profile.

    :param dir_str: string - The name of the directory and file
    :param norm: float - The scaling factor for the normalized profile
    :return: The resulting profile and its corresponding t-axis
    '''
    profile_data = np.load(dir_str)
    profile = profile_data[:, 0]
    bin = profile_data[:, 1]
    profile = norm * profile / np.sum(profile)
    return profile, bin


def IQ_induced_voltage_from_impedance(totalInducedVoltage):

    induced_voltage = totalInducedVoltage.induced_voltage

    return

def plot_induced_voltage(tracker):
    ind_volt = tracker.totalInducedVoltage.induced_voltage

    plt.figure()
    plt.title('Total induced voltage')
    plt.plot(tracker.profile.bin_centers, ind_volt)


def delta_function(x):
    out = np.zeros(x.shape)
    ind = np.argmin(np.abs(x))
    out[ind] = 1
    return out

def gaussian(x, mu, sigma):
    return (1/sigma) * np.exp(-0.5 * (x - mu)**2 / sigma**2)


def rf_current_calculation(profile, t, omega_c, charge):
    charges = profile * charge

    I_f = 2. * charges * np.cos(omega_c * t)
    Q_f = 2. * charges * np.sin(omega_c * t)

    return I_f + 1j * Q_f


def matr_conv(I, h):
    """Convolution of beam current with impulse response; uses a complete
    matrix with off-diagonal elements."""

    return fftconvolve(I, h, mode='full')[:I.shape[0]]


def rect(t):
    out = np.zeros(t.shape)
    for i in range(len(out)):
        if t[i] < 1/2 and t[i] > -1/2:
            out[i] = 1

    return out

def tri(t):
    out = np.zeros(t.shape)
    for i in range(len(out)):
        if t[i] < 0 and t[i] > -1:
            out[i] = t[i] + 1
        elif t[i] > 0 and t[i] < 1:
            out[i] = -t[i] + 1

    return out


def generator_matrix(t, domega, tau):
    '''
    Matrix elements of the generator response matrix as they are given in the I/Q Model of the SPS 200 MHz Travelling
    Wave Cavity and Feedforward Design paper by P. Baudrenghien and T. Mastordis.
    :param t: Time array
    :param domega: difference in frequency between carrier and central frequency
    :param tau: cavity filling time
    :return: Matrix elements
    '''

    hgs = (1/tau) * rect(t / tau - 0.5) * np.cos(domega * t)
    hgc = -(1/tau) * rect(t / tau - 0.5) * np.sin(domega * t)
    return hgs, hgc

def beam_matrix(t, domega , tau):
    '''
    Matrix elements of the generator response matrix as they are given in the I/Q Model of the SPS 200 MHz Travelling
    Wave Cavity and Feedforward Design paper by P. Baudrenghien and T. Mastordis.
    :param t: Time array
    :param domega: difference in frequency between carrier and central frequency
    :param tau: cavity filling time
    :return: Matrix elements
    '''

    hbs = -(1/tau) * tri(t/tau) * np.cos(domega * t) - (1/tau) * tri(t/tau) * np.sign(t/tau) * np.cos(domega * t)
    hbc = (1/tau) * tri(t/tau) * np.sin(domega * t) + (1/tau) * tri(t/tau) * np.sign(t/tau) * np.sin(domega * t)
    return hbs, hbc


def plot_OTFB_signals(OTFB, h, t_rf):
    t_c = np.linspace(0, h * t_rf, h)

    plt.figure()
    plt.title('Antenna')
    plt.plot(t_c, OTFB.V_ANT[-h:].real, color='r')
    plt.plot(t_c, OTFB.V_ANT[-h:].imag, color='b')

    plt.figure()
    plt.title('Generator')
    plt.plot(t_c, OTFB.V_IND_COARSE_GEN[-h:].real, color='r')
    plt.plot(t_c, OTFB.V_IND_COARSE_GEN[-h:].imag, color='b')


def import_measurement_signals(cavity_number, file_date, timestamp, signal_name, SHOW_SIG = False):
    r'''
    Function to import measured signals from the real SPS OTFB.

    :param cavity_number: int - Cavity number
    :param file_date: int - Date of the measurement
    :param timestamp: int - Timestamp of when the measurement was made
    :param signal_name: string - name of the signal

    :return: the signal arrays and the relevant time array
    '''
    file_dir = f'../data_files/OTFB_november_measurements/' \
               f'sps_otfb_data__all_buffers__cavity{cavity_number}__flattop__{file_date}_{timestamp}.json'

    f = open(file_dir)
    data = json.load(f)
    if SHOW_SIG:
        print(data.keys())

    sampling_rate = data['SA.TWC200_expertVcavAmp.C1-ACQ']['samplingRate'] * 1e6
    dt = 1 / sampling_rate
    N_samples = len(data['SA.TWC200_expertVcavAmp.C1-ACQ']['data'])

    t_array = np.linspace(0, dt * N_samples, N_samples)
    voltage_data = np.array(data[f'SA.TWC200_expert{signal_name}Amp.C{cavity_number}-ACQ']['data'])
    phase_data = np.array(data[f'SA.TWC200_expert{signal_name}Phase.C{cavity_number}-ACQ']['data']) * np.pi / 180
    return voltage_data, phase_data, t_array


def find_closes_value(array, value):
    r'''
    Finding the index to the closes value in an array.

    :param array: array
    :param value: float
    :return: index of the closes value
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_amp_from_linear_regression(data, dist):
    x = np.linspace(0, len(data), len(data))

    a, b, rval, pval, stderr = linregress(x, data)
    line = a * x + b
    data_wo_line = data - line
    data_wo_line_abs = np.abs(data_wo_line)
    peaks, _ = find_peaks(data_wo_line_abs, distance=dist)

    error = np.interp(x, peaks, data_wo_line_abs[peaks])
    return line, error


def retrieve_power(directory, file_names, cav_number, n_points):
    power_data = np.zeros((len(file_names), n_points))
    time_data = np.zeros((len(file_names), n_points))
    power_data_str = f'SA.TWC200_expertIcFwdPower.C{cav_number}-ACQ'

    for i in range(len(file_names)):
        with open(directory + file_names[i]) as f:
            data_i = json.load(f)

            power_data[i,:] = data_i[power_data_str]['data']
            sampling_period = 1 / (data_i[power_data_str]['samplingRate'] * 1e6)
            time_data[i,:] = np.linspace(0, sampling_period * n_points, n_points)

    return power_data, time_data

def retrieve_antenna_voltage(directory, file_names, cav_number, n_points):
    vant_data = np.zeros((len(file_names), n_points))
    time_data = np.zeros((len(file_names), n_points))
    vant_data_str = f'SA.TWC200_expertVcavAmp.C{cav_number}-ACQ'


    for i in range(len(file_names)):
        with open(directory + file_names[i]) as f:
            data_i = json.load(f)

            vant_data[i, :] = data_i[vant_data_str]['data']
            sampling_period = 1 / (data_i[vant_data_str]['samplingRate'] * 1e6)
            time_data[i, :] = np.linspace(0, sampling_period * n_points, n_points)

    return vant_data, time_data


def reshape_data(data, t, T_rev):
    N_turns = int(round(t[-1]/T_rev))
    n_points_per_turn = int(round(data.shape[1]/N_turns))
    N_shots = data.shape[0]

    data_reshaped = np.zeros((N_turns * N_shots, n_points_per_turn))
    t_turn = np.linspace(0, T_rev, n_points_per_turn)
    t_reshaped = np.zeros(data_reshaped.shape)

    k = 0
    for j in range(N_shots):
        for i in range(N_turns):
            start_ind_i = find_closes_value(t, i * T_rev)
            end_ind_i = find_closes_value(t, (i + 1) * T_rev)

            data_turn_i = data[j, start_ind_i: end_ind_i]
            t_i = t[start_ind_i: end_ind_i] - i * T_rev
            data_reshaped[k, :] = np.interp(t_turn, t_i, data_turn_i)
            t_reshaped[k, :] = t_turn
            k += 1

    return data_reshaped, t_reshaped


def plot_measurement_shots(data, t):

    plt.figure()
    for i in range(data.shape[0]):
        plt.plot(t[i,:], data[i,:])


def find_turn_by_turn_variantions(data, n_shots):
    n_turns = data.shape[0] // n_shots
    var_min = np.zeros(n_shots)
    var_max = np.zeros(n_shots)

    for i in range(n_shots):
        shot_data = data[i * n_turns: (i + 1) * n_turns, :]

        mean_shot = np.mean(shot_data, axis=0)
        max_shot = np.zeros(mean_shot.shape)
        min_shot = np.zeros(mean_shot.shape)

        for j in range(len(mean_shot)):
            max_shot[j] = np.max((shot_data[:,j] - mean_shot[j])/mean_shot[j])
            min_shot[j] = np.min((shot_data[:,j] - mean_shot[j])/mean_shot[j])

        var_max[i] = np.max(max_shot)
        var_min[i] = np.min(min_shot)

    return var_max, var_min

def find_shot_by_shot_variantions(data):
    mean_sig = np.mean(data, axis=0)
    max_dev = np.zeros(mean_sig.shape)
    min_dev = np.zeros(mean_sig.shape)

    for j in range(len(mean_sig)):
        max_dev[j] = np.max((data[:, j] - mean_sig[j]) / mean_sig[j])
        min_dev[j] = np.min((data[:, j] - mean_sig[j]) / mean_sig[j])

    return np.max(max_dev), np.min(min_dev)

def find_average_dipole_oscillation(data, nb, until_turn, distance, batch_length, number_of_batches):
    lines = np.zeros(data[:, :until_turn].shape)
    errors = np.zeros(data[:, :until_turn].shape)

    for i in range(data.shape[0]):
        data_i = (data[i, :until_turn] - nb[i]) * 1e9
        lines[i, :], errors[i, :] = find_amp_from_linear_regression(data_i, dist=distance)

    avg_dipole_osc = np.zeros((number_of_batches, errors.shape[1]))
    for i in range(number_of_batches):
        avg_dipole_osc[i, :] = np.mean(errors[i * batch_length: (i + 1) * batch_length,:], axis=0)

    return avg_dipole_osc


def get_bunch_pos_in_buckets(start_bunch, bunch_spacing, batch_spacing, number_of_batches, batch_length):

    normal_buckets = np.linspace(0, (batch_length-1) * bunch_spacing, batch_length)
    for i in range(number_of_batches - 1):
        normal_buckets = np.concatenate((normal_buckets,
                                         np.linspace(0, (batch_length - 1) * bunch_spacing, batch_length)
                                         + batch_spacing + normal_buckets[-1]))

    return normal_buckets + start_bunch




