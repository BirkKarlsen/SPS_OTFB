'''
File to perform various sanity checks of the codes that are written.

author: Birk Emil Karlsen-Bæck
'''
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt


def delta_function(t):
    out = np.zeros(t.shape, dtype=complex)
    idx = np.abs(t).argmin()
    out[idx] = 1
    return out

# Matrix convolution test
Nt = 1000

t = np.linspace(0, 3 * 2 * np.pi, Nt)
x_IQ = np.cos(t) + 1j * np.sin(t)
MAT = delta_function(t) + 1j * delta_function(t)

analytical_solution = (np.cos(t) - np.sin(t)) + 1j * (np.cos(t) + np.sin(t))

solution_by_convolution = fftconvolve(MAT, x_IQ, mode='full')[:MAT.shape[0]]

plt.plot(t, analytical_solution.real, color='r', label='real, an')
plt.plot(t, analytical_solution.imag, color='b', label='imag, an')
plt.plot(t, solution_by_convolution.real, color='r', label='real, conv', linestyle='--', marker='x')
plt.plot(t, solution_by_convolution.imag, color='b', label='imag, conv', linestyle='--', marker='x')

plt.legend()
plt.show()