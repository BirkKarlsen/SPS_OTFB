'''
File with functions for data treatment
'''

import numpy as np
import matplotlib.pyplot as plt
from blond_common.fitting.profile import binomial_amplitudeN_fit, FitOptions
from blond_common.interfaces.beam.analytic_distribution import binomialAmplitudeN
from scipy.signal import find_peaks
from scipy.stats import linregress
from scipy.interpolate import interp1d
import utility_files.analysis_tools as at
import os


def oscillation_study(sig):
    sig_max = np.max(sig)
    sig_min = np.min(sig)

    return sig_max, sig_min

def save_osc(max_arr, min_arr, sdir):

    np.save(sdir + "sim_data/max_power", max_arr)
    np.save(sdir + "sim_data/min_power", min_arr)

    plt.figure()
    plt.title("Turn-by-turn Power Oscillations")
    plt.plot(max_arr, color='r', label='max')
    plt.plot(min_arr, color='b', label='min')
    plt.legend()
    plt.xlabel('Turns')
    plt.ylabel('Power')
    plt.savefig(sdir + "fig/power_oscillations")



def save_data(OTFB, dir, i):
    r'''
    Saves the antenna voltage, induced generator and beam voltages, the generator current
    and the generator power for turn i in the directory dir.

    Parameters
    ----------
    :param OTFB: class - The SPSCavityFeedback object that contain the signals
    :param dir: string - the name of the diretory to save the data in
    :param i: int - the turn number
    '''
    # 3-section signals:
    np.save(dir + f'3sec_Vant_{i}', OTFB.OTFB_1.V_ANT[-OTFB.OTFB_1.n_coarse:])
    np.save(dir + f'3sec_power_{i}', OTFB.OTFB_1.P_GEN[-OTFB.OTFB_1.n_coarse:])
    np.save(dir + f'3sec_Igen_{i}', OTFB.OTFB_1.I_GEN[-OTFB.OTFB_1.n_coarse:])
    np.save(dir + f'3sec_Vindgen_{i}', OTFB.OTFB_1.V_IND_COARSE_GEN[-OTFB.OTFB_1.n_coarse:])
    np.save(dir + f'3sec_Vindbeam_{i}', OTFB.OTFB_1.V_IND_COARSE_BEAM[-OTFB.OTFB_1.n_coarse:])

    # 4-section signals
    np.save(dir + f'4sec_Vant_{i}', OTFB.OTFB_2.V_ANT[-OTFB.OTFB_2.n_coarse:])
    np.save(dir + f'4sec_power_{i}', OTFB.OTFB_2.P_GEN[-OTFB.OTFB_2.n_coarse:])
    np.save(dir + f'4sec_Igen_{i}', OTFB.OTFB_2.I_GEN[-OTFB.OTFB_2.n_coarse:])
    np.save(dir + f'4sec_Vindgen_{i}', OTFB.OTFB_2.V_IND_COARSE_GEN[-OTFB.OTFB_2.n_coarse:])
    np.save(dir + f'4sec_Vindbeam_{i}', OTFB.OTFB_2.V_IND_COARSE_BEAM[-OTFB.OTFB_2.n_coarse:])

def save_profile(Profile, dir, i):
    profile_data = np.zeros((Profile.n_macroparticles.shape[0], 2))
    profile_data[:, 0] = Profile.n_macroparticles
    profile_data[:, 1] = Profile.bin_centers

    np.save(dir + f'profile_{i}', profile_data)


def pos_from_fwhm(profile, t, max_pos, window, N_interp):
    max_val = profile[max_pos]
    hm = max_val / 2
    sliced_prof = profile[max_pos - window: max_pos + window]
    sliced_t = t[max_pos - window: max_pos + window]

    # Find the measurements points closes to the half-max
    left_w = find_nearest_index(sliced_prof[:window], hm)              # max_pos is absolute which is wrong
    right_w = find_nearest_index(sliced_prof[window:], hm) + max_pos
    left_prof_points = profile[left_w - 1:left_w + 2]
    left_t_points = t[left_w - 1:left_w + 2]
    right_prof_points = profile[right_w - 1:right_w + 2]
    right_t_points = t[right_w - 1:right_w + 2]

    left_t_array = np.linspace(t[left_w - 1], t[left_w + 1], N_interp)
    right_t_array = np.linspace(t[right_w - 1], t[right_w + 1], N_interp)

    left_prof_interp = np.interp(left_t_array, left_prof_points, left_t_points)
    right_prof_interp = np.interp(right_t_array, right_prof_points, right_t_points)

    left_ind = find_nearest_index(left_prof_interp, hm)
    right_ind = find_nearest_index(right_prof_interp, hm)

    return (left_t_array[left_ind] + right_t_array[right_ind]) / 2


def bunch_by_bunch_deviation(profile, t, distance=20, height=0.15, N_batch = 1, from_fwhm=False):
    dt = t[1] - t[0]
    pos, _ = find_peaks(profile, height=height, distance=distance)
    bunch_pos = t[pos]
    bunch_nr = np.linspace(1, len(pos), len(pos))
    bunch_per_batch = int(len(pos) / N_batch)

    if from_fwhm:
        for i in range(len(bunch_pos)):
            bunch_pos[i] = pos_from_fwhm(profile, t, pos[i], int(10e-9 / 2 / dt), 1000)

    bunch_pos = bunch_pos.reshape((N_batch, bunch_per_batch))
    bunch_nr = bunch_nr.reshape((N_batch, bunch_per_batch))
    fittet_line = np.zeros((N_batch, bunch_per_batch))
    bbb_dev = np.zeros((N_batch, bunch_per_batch))

    for i in range(N_batch):
        slope, intercept, r_val, p_val, std_err = linregress(bunch_nr[i,:], bunch_pos[i,:])
        fittet_line[i,:] = slope * bunch_nr[i,:] + intercept

        bbb_dev[i,:] = bunch_pos[i,:] - fittet_line[i,:]

    return bunch_pos.flatten(), fittet_line.flatten(), bbb_dev.flatten()


def import_profiles(dir, files, N_samples_per_file = 9999900):
    profile_datas = np.zeros((len(files), N_samples_per_file))
    profile_datas_corr = np.zeros((len(files), N_samples_per_file))
    n = 0
    for f in files:
        profile_datas[n,:] = np.load(dir + f + '.npy')

        conf_f = open(dir + f + '.asc', 'r')
        acq_params = conf_f.readlines()
        conf_f.close()

        delta_t = float(acq_params[6][39:-1])
        frame_length = [int(s) for s in acq_params[7].split() if s.isdigit()][0]
        N_frames = [int(s) for s in acq_params[8].split() if s.isdigit()][0]
        trigger_offsets = np.zeros(N_frames, )
        for line in np.arange(19, 119):
            trigger_offsets[line - 20] = float(acq_params[line][35:-2])

        timeScale = np.arange(frame_length) * delta_t

        # data = np.load(fullpath)
        data = np.reshape(np.load(dir + f + '.npy'), (N_frames, frame_length))
        data_corr = np.zeros((N_frames, frame_length))

        for i in range(N_frames):
            x = timeScale + trigger_offsets[i]
            A = interp1d(x, data[i,:], fill_value=0, bounds_error=False)
            data_corr[i,:] = A(timeScale)

        profile_datas_corr[n,:] = data_corr.flatten()
        n += 1

    return profile_datas, profile_datas_corr



def restack_turns(prof_array, t_array, T_rev):
    N_turns = int(t_array[-1] // T_rev)
    n_samples = int(len(t_array[np.where(t_array < T_rev)]))
    new_prof_array = np.zeros((N_turns, n_samples))
    new_t_array = np.zeros(((N_turns, n_samples)))

    for i in range(N_turns):
        new_prof_array[i,:] = prof_array[i * n_samples: (i + 1)*n_samples]
        new_t_array[i, :] = t_array[i * n_samples: (i + 1) * n_samples] - i * T_rev

    return new_prof_array, new_t_array



def find_nearest_index(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()



def find_first_bunch(prof, t, interval_t):
    interval = find_nearest_index(t, interval_t)
    max_ind = np.argmax(prof[:interval])

    return t[max_ind]



def find_first_bunch_interp(prof, t, max_t_interval, avg_t_interval, N_interp):
    # Find indices of interest
    max_interval = find_nearest_index(t, max_t_interval)
    avg_interval = find_nearest_index(t, avg_t_interval)
    prof_sliced = prof[:max_interval]
    max_ind = np.argmax(prof[:max_interval])

    # Find the half-maximum
    max_val = prof[max_ind]
    minimum_val = np.mean(prof[:avg_interval])
    half_max = (max_val - minimum_val)/2

    # Find the measurements points closes to the half-max
    left_w = find_nearest_index(prof_sliced[:max_ind], half_max)
    right_w = find_nearest_index(prof_sliced[max_ind:], half_max) + max_ind
    left_prof_points = prof[left_w - 1:left_w + 2]
    left_t_points = t[left_w - 1:left_w + 2]
    right_prof_points = prof[right_w - 1:right_w + 2]
    right_t_points = t[right_w - 1:right_w + 2]

    left_t_array = np.linspace(t[left_w - 1], t[left_w + 1], N_interp)
    right_t_array = np.linspace(t[right_w - 1], t[right_w + 1], N_interp)

    left_prof_interp = np.interp(left_t_array, left_prof_points, left_t_points)
    right_prof_interp = np.interp(right_t_array, right_prof_points, right_t_points)

    left_ind = find_nearest_index(left_prof_interp, half_max)
    right_ind = find_nearest_index(right_prof_interp, half_max)

    return (left_t_array[left_ind] + right_t_array[right_ind]) / 2




def fit_beam(prof, t, first_bunch_pos, bunch_spacing, bucket_length, batch_spacing, batch_length, N_bunches, N_interp):
    bucket_length_indices = int(bucket_length // ((t[1] - t[0]) * 2))
    bunch_spacing_indices = int(bunch_spacing // (t[1] - t[0]))
    batch_spacing_indices = int(batch_spacing // (t[1] - t[0]))

    next_peak = first_bunch_pos

    amplitudes = np.zeros(N_bunches)
    positions = np.zeros(N_bunches)
    full_lengths = np.zeros(N_bunches)
    exponents = np.zeros(N_bunches)

    for i in range(N_bunches):
        next_peak_index = find_nearest_index(t, next_peak)
        prof_i = prof[next_peak_index - bucket_length_indices: next_peak_index + bucket_length_indices]
        t_i = t[next_peak_index - bucket_length_indices: next_peak_index + bucket_length_indices]
        amplitudes[i], positions[i], full_lengths[i], exponents[i] = binomial_amplitudeN_fit(t_i, prof_i)

        if (i + 1) % batch_length == 0:
            prof_i = prof[next_peak_index + bucket_length_indices: next_peak_index + 2 * bucket_length_indices + batch_spacing_indices]
            t_i = t[next_peak_index + bucket_length_indices: next_peak_index + 2 * bucket_length_indices + batch_spacing_indices]
            if i != N_bunches - 1:
                next_peak = find_first_bunch_interp(prof_i, t_i, t_i[-1], t_i[-3 * bucket_length_indices], N_interp)

        else:
            prof_i = prof[next_peak_index + bucket_length_indices: next_peak_index + 2 * bucket_length_indices + bunch_spacing_indices]
            t_i = t[next_peak_index + bucket_length_indices: next_peak_index + 2 * bucket_length_indices + bunch_spacing_indices]
            if i != N_bunches - 1:
                next_peak = find_first_bunch_interp(prof_i, t_i, t_i[-1], t_i[-3 * bucket_length_indices], N_interp)

    return amplitudes, positions, full_lengths, exponents


def reject_outliers(data, m=2.):
    d = np.abs(data - np.mean(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def reject_outliers_2(data, m=2.):
    return data[np.abs(data - np.mean(data)) < m * np.std(data)]



def filtered_data_mean_and_std(data, m=2.):
    means = np.zeros((data.shape[1]))
    stds = np.zeros((data.shape[1]))

    for i in range(data.shape[1]):
        filtered_data_i = data[:,i]
        filtered_data_i = filtered_data_i[filtered_data_i < m]
        means[i] = np.mean(filtered_data_i)
        stds[i] = np.std(filtered_data_i)

    return means, stds



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
            if j == 71 and save_72_fits:
                x_71 = x
                y_71 = y
            if not save_72_fits:
                x_71 = x
                y_71 = y
            if len(pos) < 72:
                x_71 = x
                y_71 = y
                print("Warning - Lost bunches")
            try:
                if fit_option == 'fwhm':
                    (mu, sigma, amp) = fwhm(x, y, level=0.5)

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
                x_71 = x
                y_71 = y


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
    if not save_72_fits:
        x_71 = 0
        y_71 = 0

    return N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit, \
           Bunch_peaksFit, Bunch_Exponent, Goodness_of_fit, x_71, y_71


def getBeamPattern_4(timeScale, frames, heightFactor=0.3, distance=500, N_bunch_max=3564,
                     fit_option='fwhm', plot_fit=False, baseline_length = 0, d_interval = 1):
    dt = timeScale[1] - timeScale[0]
    fit_window = int(round(10 * 1e-9 / dt / 2))
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
        pos, _ = find_peaks(frame, height=0.015, distance=distance)
        N_bunches[i] = len(pos)
        Bunch_positions[i, 0:N_bunches[i]] = timeScale[pos]
        Bunch_peaks[i, 0:N_bunches[i]] = frame[pos]

        for j, v in enumerate(pos):
            x = 1e9 * timeScale[v - fit_window:v + fit_window]
            y = frame[v - fit_window:v + fit_window]
            baseline = np.mean(y[:baseline_length])
            y = y - baseline

            (mu, sigma, amp) = fwhm(x, y, level=0.5)
            N = int(round(4 * sigma * y.shape[0] / (2 * (x[-1] - x[0])))) + d_interval
            peak_ind = np.argmax(y)
            y = y[peak_ind - N: peak_ind + N + 1]
            x = x[peak_ind - N: peak_ind + N + 1]


            if fit_option == 'fwhm':
                (mu, sigma, amp) = fwhm(x, y, level=0.5)
        #                    (mu2, sigma2, amp2) = fwhm(x,y,level=0.95)
            else:
                (amp, mu, sigma, exponent) = binomial_amplitudeN_fit(x, y)
                y_fit = binomialAmplitudeN(x, *[amp, mu, sigma, exponent])

                if y.shape[0] != y_fit.shape[0]:
                    print(y.shape, y_fit.shape, x.shape)

                if plot_fit: #or exponent > 5:
                    print(amp, mu, sigma, exponent)

                    plt.plot(x, y, label='measurement')
                    plt.plot(x, y_fit, label='fit')
                    #plt.vlines(x[baseline_length], np.min(y), np.max(y), linestyle='--')
                    plt.legend()
                    plt.show()

                sigma /= 4

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

    return N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit, Bunch_peaksFit, Bunch_Exponent, Goodness_of_fit


def save_plots_OTFB(O, dir, i):

    # 3 section
    plt.figure()
    plt.suptitle(f'3sec, V antenna, turn {i}')
    plt.subplot(211)
    plt.plot(np.abs(O.OTFB_1.V_ANT[-O.OTFB_1.n_coarse:]), 'g', label='abs')
    plt.legend()
    plt.subplot(212)
    plt.plot(O.OTFB_1.V_ANT[-O.OTFB_1.n_coarse:].real, 'r', label='real')
    plt.plot(O.OTFB_1.V_ANT[-O.OTFB_1.n_coarse:].imag, 'b', label='imag')
    plt.legend()

    plt.savefig(dir + f'3sec_V_ANT_turn{i}')

    plt.figure()
    plt.suptitle(f'3sec, I gen, turn {i}')
    plt.subplot(211)
    plt.plot(np.abs(O.OTFB_1.I_GEN[-O.OTFB_1.n_coarse:]), 'g', label='abs')
    plt.legend()
    plt.subplot(212)
    plt.plot(O.OTFB_1.I_GEN[-O.OTFB_1.n_coarse:].real, 'r', label='real')
    plt.plot(O.OTFB_1.I_GEN[-O.OTFB_1.n_coarse:].imag, 'b', label='imag')
    plt.legend()

    plt.savefig(dir + f'3sec_I_GEN_turn{i}')

    plt.figure()
    plt.title(f'3sec, power, turn{i}')
    plt.plot(np.abs(O.OTFB_1.P_GEN[-O.OTFB_1.n_coarse:]), 'g', label='abs')
    plt.savefig(dir + f'3sec_P_GEN_turn{i}')

    plt.figure()
    plt.suptitle(f'3sec, V ind gen, turn {i}')
    plt.subplot(211)
    plt.plot(np.abs(O.OTFB_1.V_IND_COARSE_GEN[-O.OTFB_1.n_coarse:]), 'g', label='abs')
    plt.legend()
    plt.subplot(212)
    plt.plot(O.OTFB_1.V_IND_COARSE_GEN[-O.OTFB_1.n_coarse:].real, 'r', label='real')
    plt.plot(O.OTFB_1.V_IND_COARSE_GEN[-O.OTFB_1.n_coarse:].imag, 'b', label='imag')
    plt.legend()

    plt.savefig(dir + f'3sec_V_IND_GEN_turn{i}')

    plt.figure()
    plt.suptitle(f'3sec, V ind beam, turn {i}')
    plt.subplot(211)
    plt.plot(np.abs(O.OTFB_1.V_IND_COARSE_BEAM[-O.OTFB_1.n_coarse:]), 'g', label='abs')
    plt.legend()
    plt.subplot(212)
    plt.plot(O.OTFB_1.V_IND_COARSE_BEAM[-O.OTFB_1.n_coarse:].real, 'r', label='real')
    plt.plot(O.OTFB_1.V_IND_COARSE_BEAM[-O.OTFB_1.n_coarse:].imag, 'b', label='imag')
    plt.legend()

    plt.savefig(dir + f'3sec_V_IND_BEAM_turn{i}')


    # 4 section
    plt.figure()
    plt.suptitle(f'4sec, V antenna, turn {i}')
    plt.subplot(211)
    plt.plot(np.abs(O.OTFB_2.V_ANT[-O.OTFB_2.n_coarse:]), 'g', label='abs')
    plt.legend()
    plt.subplot(212)
    plt.plot(O.OTFB_2.V_ANT[-O.OTFB_2.n_coarse:].real, 'r', label='real')
    plt.plot(O.OTFB_2.V_ANT[-O.OTFB_2.n_coarse:].imag, 'b', label='imag')
    plt.legend()

    plt.savefig(dir + f'4sec_V_ANT_turn{i}')

    plt.figure()
    plt.suptitle(f'4sec, I gen, turn {i}')
    plt.subplot(211)
    plt.plot(np.abs(O.OTFB_2.I_GEN[-O.OTFB_2.n_coarse:]), 'g', label='abs')
    plt.legend()
    plt.subplot(212)
    plt.plot(O.OTFB_2.I_GEN[-O.OTFB_2.n_coarse:].real, 'r', label='real')
    plt.plot(O.OTFB_2.I_GEN[-O.OTFB_2.n_coarse:].imag, 'b', label='imag')
    plt.legend()

    plt.savefig(dir + f'4sec_I_GEN_turn{i}')

    plt.figure()
    plt.title(f'4sec, power, turn {i}')
    plt.plot(np.abs(O.OTFB_2.P_GEN[-O.OTFB_2.n_coarse:]), 'g', label='abs')

    plt.savefig(dir + f'4sec_P_GEN_turn{i}')

    plt.figure()
    plt.suptitle(f'4sec, V ind gen, turn {i}')
    plt.subplot(211)
    plt.plot(np.abs(O.OTFB_2.V_IND_COARSE_GEN[-O.OTFB_2.n_coarse:]), 'g', label='abs')
    plt.legend()
    plt.subplot(212)
    plt.plot(O.OTFB_2.V_IND_COARSE_GEN[-O.OTFB_2.n_coarse:].real, 'r', label='real')
    plt.plot(O.OTFB_2.V_IND_COARSE_GEN[-O.OTFB_2.n_coarse:].imag, 'b', label='imag')
    plt.legend()

    plt.savefig(dir + f'4sec_V_IND_GEN_turn{i}')

    plt.figure()
    plt.suptitle(f'4sec, V ind beam, turn {i}')
    plt.subplot(211)
    plt.plot(np.abs(O.OTFB_2.V_IND_COARSE_BEAM[-O.OTFB_2.n_coarse:]), 'g', label='abs')
    plt.legend()
    plt.subplot(212)
    plt.plot(O.OTFB_2.V_IND_COARSE_BEAM[-O.OTFB_2.n_coarse:].real, 'r', label='real')
    plt.plot(O.OTFB_2.V_IND_COARSE_BEAM[-O.OTFB_2.n_coarse:].imag, 'b', label='imag')
    plt.legend()

    plt.savefig(dir + f'4sec_V_IND_BEAM_turn{i}')


def save_plots_FF(O, dir, i):
    # 3 section
    plt.figure()
    plt.suptitle(f'3sec, FF I beam, turn {i}')
    plt.subplot(211)
    plt.plot(np.abs(O.OTFB_1.I_BEAM_COARSE_FF[-O.OTFB_1.n_coarse_FF:]), 'g', label='abs')
    plt.legend()
    plt.subplot(212)
    plt.plot(O.OTFB_1.I_BEAM_COARSE_FF[-O.OTFB_1.n_coarse_FF:].real, 'r', label='real')
    plt.plot(O.OTFB_1.I_BEAM_COARSE_FF[-O.OTFB_1.n_coarse_FF:].imag, 'b', label='imag')
    plt.legend()

    plt.savefig(dir + f'3sec_I_BEAM_COARSE_FF_turn{i}')

    plt.figure()
    plt.suptitle(f'3sec, V_FF_CORR, turn {i}')
    plt.subplot(211)
    plt.plot(np.abs(O.OTFB_1.V_FF_CORR[-O.OTFB_1.n_coarse_FF:]), 'g', label='abs')
    plt.legend()
    plt.subplot(212)
    plt.plot(O.OTFB_1.V_FF_CORR[-O.OTFB_1.n_coarse_FF:].real, 'r', label='real')
    plt.plot(O.OTFB_1.V_FF_CORR[-O.OTFB_1.n_coarse_FF:].imag, 'b', label='imag')
    plt.legend()

    plt.savefig(dir + f'3sec_V_FF_CORR_turn{i}')

    # 4 section
    plt.figure()
    plt.suptitle(f'4sec, FF I beam, turn {i}')
    plt.subplot(211)
    plt.plot(np.abs(O.OTFB_2.I_BEAM_COARSE_FF[-O.OTFB_2.n_coarse_FF:]), 'g', label='abs')
    plt.legend()
    plt.subplot(212)
    plt.plot(O.OTFB_2.I_BEAM_COARSE_FF[-O.OTFB_2.n_coarse_FF:].real, 'r', label='real')
    plt.plot(O.OTFB_2.I_BEAM_COARSE_FF[-O.OTFB_2.n_coarse_FF:].imag, 'b', label='imag')
    plt.legend()

    plt.savefig(dir + f'4sec_I_BEAM_COARSE_FF_turn{i}')

    plt.figure()
    plt.suptitle(f'4sec, V_FF_CORR, turn {i}')
    plt.subplot(211)
    plt.plot(np.abs(O.OTFB_2.V_FF_CORR[-O.OTFB_2.n_coarse_FF:]), 'g', label='abs')
    plt.legend()
    plt.subplot(212)
    plt.plot(O.OTFB_2.V_FF_CORR[-O.OTFB_2.n_coarse_FF:].real, 'r', label='real')
    plt.plot(O.OTFB_2.V_FF_CORR[-O.OTFB_2.n_coarse_FF:].imag, 'b', label='imag')
    plt.legend()

    plt.savefig(dir + f'4sec_V_FF_CORR_turn{i}')


def bunch_params(profile, get_72 = True):
    gen_prof = np.array([profile.n_macroparticles])
    N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit, \
    Bunch_peaksFit, Bunch_Exponent, Goodness_of_fit, x_71, y_71 \
        = getBeamPattern_3(profile.bin_centers, gen_prof.T,
                           distance=2**7 * 3, fit_option='fwhm', heightFactor=50,
                           wind_len=5, save_72_fits=get_72)
    return Bunch_lengths[0,:], Bunch_positions[0,:], Bunch_positionsFit[0,:] * 1e-9, x_71 * 1e-9, y_71


def plot_fit_per_turn(x, y, pos, pos_fit, bl, sdir, i, PLOT = False):
    rr = 1.5
    plt.figure()
    plt.title(f"Bunch 72, at turn {i}")
    plt.plot(x, y, color='r')
    plt.vlines(pos, np.max(y) * rr, np.min(y) * rr, color='black')
    plt.vlines(pos - bl/2 * 1e-9, np.max(y) * rr, np.min(y) * rr, color='b')
    plt.vlines(pos + bl/2 * 1e-9, np.max(y) * rr, np.min(y) * rr, color='b')
    plt.vlines(pos_fit, np.max(y) * rr, np.min(y) * rr, color='black', linestyles='--')
    plt.vlines(pos_fit - bl/2 * 1e-9, np.max(y) * rr, np.min(y) * rr, color='b', linestyles='--')
    plt.vlines(pos_fit + bl/2 * 1e-9, np.max(y) * rr, np.min(y) * rr, color='b', linestyles='--')
    if PLOT:
        plt.show()
    else:
        plt.savefig(sdir + f"fig/bunch_72_fit_turn_{i}")



def plot_params(fwhm, pos, pos_fit, pow, vant, sdir, t_rf, i, n, MB = True):
    x = np.linspace(0, i, n)
    plt.figure()
    plt.title("FWHM, first and final per batch")
    plt.plot(x, fwhm[0,:n], label="bu1ba1")
    plt.plot(x, fwhm[71,:n], label="bu72ba1")
    if MB:
        plt.plot(x, fwhm[72,:n], label="bu1ba2")
        plt.plot(x, fwhm[143,:n], label="bu72ba2")
        plt.plot(x, fwhm[144,:n], label="bu1ba3")
        plt.plot(x, fwhm[215,:n], label="bu72ba3")
        plt.plot(x, fwhm[216,:n], label="bu1ba4")
        plt.plot(x, fwhm[287,:n], label="bu72ba4")
    plt.legend()
    plt.savefig(sdir + "fig/FWHM_tbt")

    plt.figure()
    plt.title(r"$\Delta t$, first and final per batch")
    plt.plot(x, pos[0, :n], label="bu1ba1")
    plt.plot(x, pos[71, :n] - 5 * 71 * t_rf, label="bu72ba1")
    if MB:
        plt.plot(x, pos[72, :n] - (5 * 71 + 50) * t_rf, label="bu1ba2")
        plt.plot(x, pos[143, :n] - (5 * 142 + 50) * t_rf, label="bu72ba2")
        plt.plot(x, pos[144, :n] - (5 * 142 + 100) * t_rf, label="bu1ba3")
        plt.plot(x, pos[215, :n] - (5 * 213 + 100) * t_rf, label="bu72ba3")
        plt.plot(x, pos[216, :n] - (5 * 213 + 150) * t_rf, label="bu1ba4")
        plt.plot(x, pos[287, :n] - (5 * 284 + 150) * t_rf, label="bu72ba4")
    plt.legend()
    plt.savefig(sdir + "fig/pos_tbt")

    plt.figure()
    plt.title(r"$\Delta t$, first and final per batch")
    plt.plot(x, pos_fit[0, :n], label="bu1ba1")
    plt.plot(x, pos_fit[71, :n] - (5 * 71) * t_rf, label="bu72ba1")
    if MB:
        plt.plot(x, pos_fit[72, :n] - (5 * 71 + 50) * t_rf, label="bu1ba2")
        plt.plot(x, pos_fit[143, :n] - (5 * 142 + 50) * t_rf, label="bu72ba2")
        plt.plot(x, pos_fit[144, :n] - (5 * 142 + 100) * t_rf, label="bu1ba3")
        plt.plot(x, pos_fit[215, :n] - (5 * 213 + 100) * t_rf, label="bu72ba3")
        plt.plot(x, pos_fit[216, :n] - (5 * 213 + 150) * t_rf, label="bu1ba4")
        plt.plot(x, pos_fit[287, :n] - (5 * 284 + 150) * t_rf, label="bu72ba4")
    plt.legend()
    plt.savefig(sdir + "fig/pos_fit_tbt")

    plt.figure()
    plt.title("Max power")
    plt.plot(x, pow[0, :n], label='3sec')
    plt.plot(x, pow[1, :n], label='4sec')
    plt.legend()
    plt.savefig(sdir + "fig/max_pow_tbt")

    plt.figure()
    plt.title("Max Antenna voltage")
    plt.plot(x, vant[0, :n], label='3sec')
    plt.plot(x, vant[1, :n], label='4sec')
    plt.legend()
    plt.savefig(sdir + "fig/max_vant_tbt")


def save_params(fwhm, pos, pos_fit, pow, vant, sdir):
    np.save(sdir + "sim_data/fwhm_tbt", fwhm)
    np.save(sdir + "sim_data/pos_tbt", pos)
    np.save(sdir + "sim_data/pos_fit_tbt", pos_fit)
    np.save(sdir + "sim_data/max_pow_tbt", pow)
    np.save(sdir + "sim_data/max_vant_tbt", vant)


def plot_ramp(intensity, i, n, sdir):
    x = np.linspace(0, i, n)
    plt.figure()
    plt.title(r"Intensity per turn")
    plt.plot(x, intensity[:n], color='r')
    plt.xlabel(r"Turns")
    plt.ylabel(r"Intensity")
    plt.savefig(sdir + "fig/intensity_tbt")


def plot_induced_voltage(tracker, total_ind):
    pass


def plot_bbb_offset(pos_fit, N_batches, sdir, i, show=False):
    plt.figure()
    plt.title(f'bunch-by-bunch offset, turn {i}')
    pos_fit = pos_fit.reshape((N_batches, 72))

    for j in range(N_batches):
        bbb_offset = at.find_offset(pos_fit[j,:])
        x = np.linspace(0, len(pos_fit[j,:]), len(pos_fit[j,:]))
        plt.plot(x, bbb_offset * 1e9, label=f'ba{j + 1}')

    plt.xlabel('Bunch Number')
    plt.ylabel('Offset [ns]')
    plt.legend()
    plt.savefig(sdir + f"bbb_offset_{i}")
    if show:
        plt.show()
    else:
        plt.savefig(sdir + f"bbb_offset_{i}")


def find_induced_and_generator(OTFB, rfstation, profile, tracker):
    # Beam-induced voltage
    ind_amp = np.abs(OTFB.OTFB_1.V_IND_FINE_BEAM[-profile.n_slices:] +
                     OTFB.OTFB_2.V_IND_FINE_BEAM[-profile.n_slices:])
    ind_phase = np.angle(OTFB.OTFB_1.V_IND_FINE_BEAM[-profile.n_slices:] +
                         OTFB.OTFB_2.V_IND_FINE_BEAM[-profile.n_slices:]) - np.pi/2

    beam_induced = ind_amp * np.sin(rfstation.omega_rf[0, tracker.counter] * profile.bin_centers
                                    + rfstation.phi_rf[0, tracker.counter] + ind_phase)

    # Generator-induced voltage
    ind_amp = np.abs(OTFB.V_sum - OTFB.OTFB_1.V_IND_FINE_BEAM[-profile.n_slices:]
                     - OTFB.OTFB_2.V_IND_FINE_BEAM[-profile.n_slices:])
    ind_phase = np.angle(OTFB.V_sum - OTFB.OTFB_1.V_IND_FINE_BEAM[-profile.n_slices:]
                         - OTFB.OTFB_2.V_IND_FINE_BEAM[-profile.n_slices:]) - np.pi/2

    generator_induced = ind_amp * np.sin(rfstation.omega_rf[0, tracker.counter] * profile.bin_centers
                                    + rfstation.phi_rf[0, tracker.counter] + ind_phase)

    return beam_induced, generator_induced

def find_effective_induced(OTFB, rfstation, profile, tracker, n_phi = 100):
    # From tracker
    rf_voltage = tracker.rf_voltage

    # Voltage without OTFB
    rf_wo = tracker.voltage[0, tracker.counter] * np.sin(rfstation.omega_rf[0, tracker.counter] * profile.bin_centers
                                                         + rfstation.phi_rf[0, tracker.counter])

    return rf_voltage - rf_wo


def file_names_in_dir_from_prefix(drt, prefix):
    name_list = []

    for file in os.listdir(drt[:-1]):
        if file.startswith(prefix):
            name_list.append(file)

    return name_list


def mk_file_names_volt_scan(volts, prefix):
    file_names = []

    for i in range(len(volts)):
        suffix = f'_fr1_vc1_ve{100 * volts[i]:.0f}_bl100_llrf_20.npy'
        file_names.append(prefix + suffix)

    return np.array(file_names)

def get_data_from_files(dir_folder, file_names):

    # Import test data to know the length of the arrays
    td = np.load(dir_folder + file_names[0])
    data_shape = td.shape

    data = np.zeros((len(file_names), data_shape[0], data_shape[1]))

    for i in range(len(file_names)):
        data[i,:,:] = np.load(dir_folder + file_names[i])

    return data
