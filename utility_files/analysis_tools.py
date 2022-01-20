'''
Tools to analyse data

author: Birk Emil Karlsen-Bæck
'''
import matplotlib.pyplot as plt
import numpy as np
from blond_common.fitting.profile import binomial_amplitudeN_fit, FitOptions
from blond_common.interfaces.beam.analytic_distribution import binomialAmplitudeN
from scipy.signal import find_peaks
from scipy.stats import linregress
from scipy.interpolate import interp1d


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

    fit_line = sl * x + inter

    print(sl)

    offset_fit1 = pos1 - fit_line
    offset_fit2 = pos2 - fit_line
    return offset_fit1, offset_fit2


def find_offset_trf(pos1, pos2, t_rf):
    x = np.linspace(0, len(pos1), len(pos1))

    line = 5 * t_rf * x + pos1[0]

    print(5 * t_rf)

    offset1 = pos1 - line
    offset2 = pos2 - line

    return offset1, offset2



def plot_IQ(Va, Vg, Vb, titstr = '', start=1000, end=3040, norm = False, wind = 3.1e6):
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