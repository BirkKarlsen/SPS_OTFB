'''
File to show plots for the FF design section

Author: Birk Emil Karlsen-Bæck
'''

# Options -------------------------------------------------------------------------------------------------------------
PLT_BEAM_LOADING = True

# Imports -------------------------------------------------------------------------------------------------------------
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
import utility_files.analysis_tools as at
import utility_files.data_utilities as dut
from scipy.signal import fftconvolve

from blond.llrf.signal_processing import feedforward_filter_TWC3_1, feedforward_filter_TWC4_1
from blond.llrf.impulse_response import SPS3Section200MHzTWC, SPS4Section200MHzTWC

plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 16
    })


# Impulse response ----------------------------------------------------------------------------------------------------

if PLT_BEAM_LOADING:
    n_points = 2000
    step = np.zeros(n_points)
    step[-n_points//2:] = 1

    plt.figure()
    plt.plot(step)

    T = 461.91831e-9
    b = 0.1  # tau DeltaF
    x = np.linspace(-0.5, 1.5, 1000)
    t = np.linspace(-2, 2, 1000)
    dt = t[1] - t[0]

    hbs_even, hbs_odd = at.beam_even_odd(t * T, 2 * np.pi * b / T, T)

    plt.figure()
    plt.plot(hbs_even)
    plt.plot(hbs_odd)

    even_response = fftconvolve(step, hbs_even, mode='full')[:step.shape[0]][-hbs_even.shape[0]:] * dt * T
    odd_response = fftconvolve(step, hbs_odd, mode='full')[:step.shape[0]][-hbs_odd.shape[0]:] * dt * T

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    fig.suptitle(r'Step Function Beam Loading, $h_{b,s}$')

    ax[0].set_title('Even part')
    ax[0].plot(t * T * 1e6, even_response, color='r')
    ax[0].set_xlabel(r'Time [$\mu$s]')
    ax[0].set_ylabel(r'$V / V_m$ [-]')
    ax[0].grid()

    ax[1].set_title('Odd part')
    ax[1].plot(t * T * 1e6, odd_response, color='r')
    ax[1].set_xlabel(r'Time [$\mu$s]')
    ax[1].set_ylabel(r'$V / V_m$ [-]')
    ax[1].grid()

    plt.show()






