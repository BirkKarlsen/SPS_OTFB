'''
File to make the open-loop bode and Nyquist plots of the SPS OTFB.

Author: Birk Emil Karlsen-Bæck
'''

# Imports ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import utility_files.data_utilities as dut
import utility_files.analysis_tools as at
import network_analysis_functions as naf


# Parameters ------------------------------------------------------------------
h = 4620                                                # Harmonic number
omega_rf = 2 * np.pi * 200.394e6                        # RF Angular Frequency
omega_rev = omega_rf / h                                # Revolution Frequency
domega = omega_rev / 2                                  # Frequency Sweep range
n_samples = 20
m_samples = 1
k = 1
m0 = 0
m_jump = 1

t_max = 2 * np.pi / (domega / n_samples)
T_rev = 2 * np.pi / omega_rev
n_turns = int(t_max / T_rev) + 1
print(f'Have to track {n_turns} turns for each frequency')
print(f't_max is {t_max} s and revolution frequency is {T_rev} s')
print(f'Corresponding to {1 / t_max} Hz and {1 / T_rev} Hz')

# Define Arrays ---------------------------------------------------------------
signal = np.zeros(h * n_turns, dtype=complex)
output = np.zeros(h * n_turns, dtype=complex)

input_freq = np.zeros(n_samples * m_samples)
ind = 0
for j in range(m_samples):
    for i in range(n_samples):
        input_freq[ind] = k * omega_rf + (m_jump * j + m0) * omega_rev + 2 * domega * (i / n_samples - 1/2)
        ind += 1

output_response_corr = np.zeros(n_samples * m_samples, dtype=complex)
output_response_vec = np.zeros(n_samples * m_samples, dtype=complex)

#input_freq = np.array([omega_rf])
t = np.linspace(0, n_turns * T_rev, n_turns * h)

#plt.figure()
#plt.plot(t[:h * finer_grid], np.sin(omega_rf * t)[:h * finer_grid], label=f'{omega_rf}')
#plt.plot(t[:h * finer_grid], np.sin(omega_rev * t)[:h * finer_grid], label=f'{omega_rev}')
#plt.legend()
#plt.show()


# Network Analysis ------------------------------------------------------------

print(f'Commence Analysis...')

for i in range(len(input_freq)):
    if i % (n_samples/10) == 0:
        print(f'Turn {i}')

    OTFB = naf.init_OTFB(df=0 * 356e3)
    signal = naf.convert_to_IQ(input_freq[i], omega_rf, t)
    signal1 = naf.generate_sinusoid(input_freq[i] - omega_rf, t)


    for j in range(n_turns):
        OTFB.NOISE[:h] = OTFB.NOISE[-h:]
        OTFB.NOISE[-h:] = 100 * signal[j * h:(j + 1) * h]

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