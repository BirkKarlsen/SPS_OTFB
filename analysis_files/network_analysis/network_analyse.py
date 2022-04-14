'''
File to make the open-loop bode and Nyquist plots of the SPS OTFB.

Author: Birk Emil Karlsen-Bæck
'''

# Imports ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as nfft
import scipy.signal as sig

plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 16
    })

import utility_files.data_utilities as dut
import utility_files.analysis_tools as at
import network_analysis_functions as naf

from blond.llrf.transfer_function import TransferFunction


# Parameters ------------------------------------------------------------------
h = 4620                                                # Harmonic number
omega_rf = 2 * np.pi * 200.394e6                        # RF Angular Frequency
omega_rev = omega_rf / h                                # Revolution Frequency
domega = omega_rev / 2                                  # Frequency Sweep range
f_rf = omega_rf / (2 * np.pi)
f_rev = omega_rev / (2 * np.pi)
#df = 356e3
#df = 0.18433333e6
#df = 62333.333
df = 0

n_samples = 20
m_samples = 1
k = 1
m0 = 0
m_jump = 1

NOISE_TYPE = 'white'

# White noise
noise_amp = 0.5 * 1e6
set_point = 1e6
noise_length = 2000

# Sine noise
sine_amp = 100

t_max = 2 * np.pi / (domega / n_samples)
T_rev = 2 * np.pi / omega_rev
if NOISE_TYPE == 'sine':
    n_turns = int(t_max / T_rev) + 1
elif NOISE_TYPE == 'white':
    n_turns = noise_length

print(f'Have to track {n_turns} turns for each frequency')
print(f't_max is {t_max} s and revolution frequency is {T_rev} s')
print(f'Corresponding to {1 / t_max} Hz and {1 / T_rev} Hz')

# Define Arrays ---------------------------------------------------------------
signal = np.zeros(h * n_turns, dtype=complex)
output = np.zeros(h * n_turns, dtype=complex)

input_freq = np.zeros(n_samples * m_samples)
if NOISE_TYPE == 'sine':
    ind = 0
    for j in range(m_samples):
        for i in range(n_samples):
            input_freq[ind] = k * omega_rf + (m_jump * j + m0) * omega_rev + 2 * domega * (i / n_samples - 1/2)
            ind += 1

output_response_corr = np.zeros(n_samples * m_samples, dtype=complex)
output_response_vec = np.zeros(n_samples * m_samples, dtype=complex)

# Network Analysis ------------------------------------------------------------

print(f'Commence Analysis...')
t = np.linspace(0, n_turns * T_rev, n_turns * h)

if NOISE_TYPE == 'white':
    input_freq = np.zeros(1)


for i in range(len(input_freq)):
    if i % (n_samples/10) == 0:
        print(f'Turn {i}')

    OTFB = naf.init_OTFB(df=df, set_point=set_point, n_pretrack=1000, G_tx=1, G_llrf=20,
                         n_sections=4)
    signal = sine_amp * naf.convert_to_IQ(input_freq[i], omega_rf, t)
    signal1 = naf.generate_sinusoid(input_freq[i] - omega_rf, t)

    if NOISE_TYPE == 'white':
        np.random.seed(1234)
        r1 = np.random.random_sample(len(t))
        np.random.seed(1234 + 1)
        r2 = np.random.random_sample(len(t))
        signal = noise_amp * np.exp(2 * np.pi * 1j * r1) * np.sqrt(-2 * np.log(r2))

    for j in range(n_turns):
        OTFB.NOISE[:h] = OTFB.NOISE[-h:]
        OTFB.NOISE[-h:] = signal[j * h:(j + 1) * h]

        OTFB.update_variables()
        OTFB.TWC.impulse_response_gen(OTFB.omega_c, OTFB.rf_centers)
        OTFB.error_and_gain()
        OTFB.comb()
        OTFB.one_turn_delay()
        OTFB.mod_to_fr()
        OTFB.mov_avg()
        OTFB.mod_to_frf()
        OTFB.sum_and_gain()
        OTFB.gen_response()

        output[j * h:(j + 1) * h] = OTFB.V_IND_COARSE_GEN[-h:]


    #plt.figure()
    #plt.subplot(211)
    #plt.plot(signal.real, color='r')
    #plt.plot(signal.imag, color='b')
    #plt.subplot(212)
    #plt.plot(output.real, color='r')
    #plt.plot(output.imag, color='b')
    #plt.show()

    output_response_corr[i] = naf.calculate_freq_response_by_correlation(signal, output)
    output_response_vec[i] = naf.calcualate_freq_response_vectorial(signal, output, input_freq[i] / (2 * np.pi))


if NOISE_TYPE == 'white':

    TF = TransferFunction(signal, output, T_s= 2 * np.pi / omega_rf, plot=False)
    TF.analyse(data_cut=0)
    n_harm = 5

    central_ind = dut.find_nearest_index(TF.f_est, 0)

    TF.H_est[central_ind] = 0
    TF.H_est[central_ind + 1] = 0
    TF.H_est[central_ind - 1] = 0

    max_val_ind = np.argmax(TF.H_est)
    #dff = TF.f_est[max_val_ind]
    #dff = -356e3 + df
    dff = -399e3 + df
    ind_min = dut.find_nearest_index(TF.f_est, -n_harm * f_rev + dff)
    ind_max = dut.find_nearest_index(TF.f_est, n_harm * f_rev + dff)
    max_val_ind = dut.find_nearest_index(TF.f_est, dff)

    f = TF.f_est[ind_min:ind_max]
    H = TF.H_est[ind_min:ind_max]

    print(TF.f_est[max_val_ind])

    plt.figure()
    plt.subplot(211)
    plt.plot(f, np.log(np.abs(H)))
    plt.subplot(212)
    plt.plot(f, np.angle(H))

    plt.figure()
    plt.subplot(211)
    plt.plot(f[:max_val_ind - ind_min], np.log(np.abs(H[:max_val_ind - ind_min])))
    plt.plot(f[max_val_ind - ind_min - 1::-1], np.log(np.abs(H[-(max_val_ind - ind_min):])))
    plt.subplot(212)
    plt.plot(f[:max_val_ind - ind_min], np.angle(H[:max_val_ind - ind_min]))
    plt.plot(f[max_val_ind - ind_min - 1::-1], np.angle(H[-(max_val_ind - ind_min):]))


    plt.figure(figsize=(6, 6))
    plt.title('$f_{rf} = 200.394$ MHz, $f_{r} = 200.038$ MHz')
    n_points = int((ind_max - ind_min) / (2 * n_harm))
    ddomega = 2 * np.pi * (200.038 + df)  - omega_rf
    #vec = 20 * (np.cos(ddomega * OTFB.TWC.tau / 2) + 1j * np.sin(ddomega * OTFB.TWC.tau / 2))
    #plt.plot([0, vec.real], [0, vec.imag])
    for i in range(2 * n_harm):
        plt.plot(H.real[i * n_points: (i + 1) * n_points],
                 H.imag[i * n_points: (i + 1) * n_points])
    plt.xlim((-25, 25))
    plt.ylim((-25, 25))
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.grid()



if NOISE_TYPE == 'sine':
    plt.figure()
    for i in range(m_samples):
        plt.plot((input_freq[i * n_samples:(i + 1) * n_samples] - omega_rf) / (2 * np.pi),
                 np.abs(output_response_corr[i * n_samples:(i + 1) * n_samples]))
    plt.grid()

    plt.figure()
    for i in range(m_samples):
        plt.plot((input_freq[i * n_samples:(i + 1) * n_samples] - omega_rf) / (2 * np.pi),
                 np.angle(output_response_corr[i * n_samples:(i + 1) * n_samples]))
    plt.grid()




    plt.figure()
    plt.title('Correlation')
    for i in range(m_samples):
        plt.plot(output_response_corr.real[i * n_samples:(i + 1) * n_samples],
                 output_response_corr.imag[i * n_samples:(i + 1) * n_samples], '.')
    plt.grid()

    plt.figure()
    plt.title('Vector')
    for i in range(m_samples):
        plt.plot(output_response_vec.real[i * n_samples:(i + 1) * n_samples],
                 output_response_vec.imag[i * n_samples:(i + 1) * n_samples], '.')
    plt.grid()



plt.show()