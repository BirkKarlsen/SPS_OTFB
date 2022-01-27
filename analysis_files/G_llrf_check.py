import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as spfft
import scipy.constants as spc
import scipy.signal as spsig

from blond.impedances.impedance_sources import TravelingWaveCavity


plt.rcParams.update({
    'text.usetex':True,
    'text.latex.preamble':r'\usepackage{fourier}',
    'font.family':'serif',
    'font.size': 16
})

# Functions -------------------------------------------------------------------
def Gaussian_Batch(t, N_b, a, N_p, sigma, t_start):
    func = np.zeros(t.shape[0])
    for i in range(N_b):
        func += (N_p / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(1/2) * ((t - i * a - t_start)**2)/(sigma**2))
    return func

def Fourier_Gaussian_Batch(freq, N_b, a, N_p, sigma, t_start):
    func = np.zeros(freq.shape[0], dtype=complex)
    for i in range(N_b):
        func += np.exp(1j * 2 * np.pi * freq * (i * a + t_start))
    return func * (N_p / np.sqrt(2 * np.pi)) * np.exp(-(1/2) * (2 * np.pi * freq * sigma)**2)


# Travelling Wave Cavity ------------------------------------------------------
twc = TravelingWaveCavity(485202, 200.03766667e6, 462e-9)
twc_2 = TravelingWaveCavity(876112, 199.9945e6, 621e-9)


# Wake Function
h = 4620
N_time = 200 * h
turn_time = 2.30545e-5
time_array = np.linspace(0, turn_time, N_time)
twc.wake_calc(time_array)
twc_2.wake_calc(time_array)
wake = twc.wake


# Beam ------------------------------------------------------------------------
N_bunches = 72
a = 25e-9
N_p = 1.17e11
sigma = 1.87e-9 / 4
t_start = 5e-9 * 100

batch = Gaussian_Batch(time_array, N_bunches, a, N_p, sigma, t_start)
numerical_fourier_batch = spfft.fftshift(spfft.fft(batch))

freq2 = spfft.fftshift(spfft.fftfreq(N_time, time_array[1]-time_array[0]))
fourier_batch = Fourier_Gaussian_Batch(freq2, N_bunches, a, N_p, sigma, t_start)
numerical_fourier_batch = numerical_fourier_batch / np.sqrt(2 * np.pi)

twc.imped_calc(freq2)
twc_2.imped_calc(freq2)
N_half = len(numerical_fourier_batch)//2

df = freq2[1] - freq2[0]
dt = time_array[1] - time_array[0]
print("Frequency resolution:", df)
print("Time resolution:", dt)



V_beam_from_fourier = spfft.ifftshift(spfft.ifft(fourier_batch / dt * (4 * twc.impedance +
                                                                            2 * twc_2.impedance)))
V_beam_from_conv = -spsig.convolve(batch, 4 * wake + 2 * twc_2.wake, mode='full')[:len(batch)]

G_llrf = 20
plt.figure()
plt.title("Induced voltage reduction")
plt.plot(time_array * 1e6, V_beam_from_conv * spc.e * dt * 1e-6,
         color='r', label='Full')
plt.plot(time_array * 1e6, V_beam_from_conv * spc.e * dt * 1e-6 / G_llrf,
         color='b', linestyle='--', label='Reduced')
plt.xlabel(r'$\Delta t$ [$\mu$s]')
plt.ylabel(r'$V_{\textrm{ind}}$ [MV]')
plt.xlim((0, 3))
plt.legend()

print('Induced voltage:', np.max(V_beam_from_conv * spc.e * dt * 1e-6))
print('Reduced:', np.max(V_beam_from_conv * spc.e * dt * 1e-6 / G_llrf))

plt.figure()
plt.title("Cavity impedance")
plt.plot(freq2 * 1e-6, (4 * twc.impedance + 2 * twc_2.impedance).real * 1e-6,
         color='r', label='Real')
plt.plot(freq2 * 1e-6, (4 * twc.impedance + 2 * twc_2.impedance).imag * 1e-6,
         color='b', label='Imaginary')
plt.xlim((0, 2 * 200.1))
plt.xlabel(r'Frequency [MHz]')
plt.ylabel(r'Impedance [M$\Omega$]')
plt.legend()
plt.show()
