'''
This is a test file to understand and make simple calculations in preparation for the SPS FF model

Author: Birk Emil Karlsen-Bæck
'''

PLT_IMPULSE = False                 # Plot impulse response matrix elements
PLT_BEAM_LOADING = False            # Plot beam loading due to real and imaginary part of impedance
PLT_DESIRED_VEC = False             # Plot the desired vectors
PLT_OPT_FIR = False                 # Plot the optimal FIR filters when splitting real and imaginary parts
PLT_REC_SIG = False                 # Plot reconstructed signal using the FIR filters with splitting
PLT_OPT_FIR_WO = False              # Plot the optimal FIR filter without splitting the real and imaginary parts
PLT_REC_SIG_WO = False              # plot the reconstructed signals without splitting the real and imaginary parts
SECTION = 4

# Imports ---------------------------------------------------------------------
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
import utility_files.analysis_tools as at

from blond.llrf.signal_processing import feedforward_filter_TWC3, feedforward_filter_TWC4
from blond.llrf.impulse_response import SPS3Section200MHzTWC, SPS4Section200MHzTWC

plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 16
    })

if SECTION == 3:
    TWC = SPS3Section200MHzTWC()
    print(TWC.tau)
    print(len(feedforward_filter_TWC3))
else:
    TWC = SPS4Section200MHzTWC()
    print(TWC.tau)
    print(len(feedforward_filter_TWC4))

# Philips FIR filter coefficients ---------------------------------------------
phopteven = np.array([8.67639e-14, -8.32667e-15, -1.59983e-13, 1.15907e-13,
                      -2.81997e-14, -3.70814e-14, 0.000432526, 0.00302768,
                      0.00346021, 0.00346021, 0.00346021, 0.00346021,
                      0.00346021, 0.00346021, 0.00346021, 0.00346021,
                      0.00346021, 0.00346021, 0.00346021, 0.00346021,
                      0.00346021, 0.00346021, 0.00346021, 0.00302768,
                      0.000432526, -3.70814e-14, -2.81997e-14, 1.15907e-13,
                      -1.59983e-13, -8.32667e-15, 8.67639e-14])

phoptodd = np.array([-0.0140479, 0.0221217, 0.00230681, 0.00230681,
                     0.00230681, 0.00230681, -0.0177336, -0.0203287,
                     -0.0011534, -0.0011534, -0.0011534, -0.0011534,
                     -0.0110608, 0.00702393, -3.95549e-15, 0,
                     3.95549e-15, -0.00702393, 0.0110608, 0.0011534,
                     0.0011534, 0.0011534, 0.0011534, 0.0203287,
                     0.0177336, -0.00230681, -0.00230681, -0.00230681,
                     -0.00230681, -0.0221217, 0.0140479])



# Parameters for the FF -------------------------------------------------------
fs = 40.0444e6                          # Sampling frequency, was 124/4 MS/s
Ts = 1 / fs
Trev = 924 / fs                         # Revolution period, was 256/fs
Nrev = Trev * fs
if SECTION == 3:
    T = 461.91831e-9                              # Filling time [s]
    Lfilling = round(T * fs)                # Cavity response length
    Ntap = 31                               # Filter response length. MUST BE ODD
else:
    T = 620.70273e-9  # Filling time [s]
    Lfilling = round(T * fs)  # Cavity response length
    Ntap = 37  # Filter response length. MUST BE ODD


# FIR filter ------------------------------------------------------------------
def FIR(z, H):                          # z-transform of FIR
    output = 0
    for i in range(len(H)):
        output += H[i] * z**(-(i - 1))
    return output


# Impulse response ------------------------------------------------------------
b = 0.1                                 # tau DeltaF
x = np.linspace(-0.5, 1.5, 1000)
t = np.linspace(-2, 2, 1000)
hgs, hgc = at.generator_matrix(x * T, 2 * np.pi * b / T, T)
hbs, hbc = at.beam_matrix(t * T, 2 * np.pi * b / T, T)

if PLT_IMPULSE:
    plt.figure('Generator Impulse Response')
    plt.title('Generator Impulse Response')
    plt.plot(x, hgs * T)
    plt.plot(x, -hgc * T)

    plt.figure('Beam Impulse Response')
    plt.title('Beam Impulse Response')
    plt.plot(t, -hbs * T)
    plt.plot(t, hbc * T)


# Response Matrices -----------------------------------------------------------
# Step response of FIR. (M1, N1) must be odd
def Smatrix(M1, N1):
    output = np.zeros((M1, N1))
    for i in range(M1):
        for j in range(N1):
            if i - j >= (M1 - N1)/2:
                output[i,j] = 1
    return output

# Response of symmetrix rectangle of length L. P must be odd
def Rmatrix(P, L):
    output = np.zeros((P, P + L - 1))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if i - j <= 0 and i - j > -L:
                output[i,j] = 1
    return output

# Causal response of symmetric rectangle of length L. P must be odd
def RCmatrix(P, L):
    output = np.zeros((P, P + L - 1))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if i - j >= 0 and i - j < L:
                output[i,j] = 1
    return output

# Impulse response
def Imatrix(P, L):
    output = np.zeros((P, L))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if i - j == 0:
                output[i,j] = 1
    return output


# Beam Loading Impedance ------------------------------------------------------
# Real part
def V_real(t, tauf):
    output = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] < -tauf:
            output[i] = 0
        elif t[i] < 0:
            output[i] = (t[i] / tauf + 1)**2 / 2
        elif t[i] < tauf:
            output[i] = -1/2 * (t[i] / tauf)**2 + t[i]/tauf + 1/2
        else:
            output[i] = 1
    return output

# Imaginary part
def V_imag(t, tauf):
    output = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] < -tauf:
            output[i] = 0
        elif t[i] < 0:
            output[i] = -(t[i] / tauf + 1) ** 2 / 2
        elif t[i] < tauf:
            output[i] = -1 / 2 * (t[i] / tauf) ** 2 + t[i] / tauf - 1 / 2
        else:
            output[i] = 0
    return output

def V_tot(t, tauf):
    return V_real(t, tauf) + V_imag(t, tauf)


t = np.linspace(-2 * T, 2 * T, 1000)
Veven = V_real(t, T)
Vodd = V_imag(t, T)


if PLT_BEAM_LOADING:
    plt.figure('Voltage due to real part of impedance')
    plt.title('Voltage due to real part of impedance')
    plt.plot(t, Veven)

    plt.figure('Voltage due to imaginary part of impedance')
    plt.title('Voltage due to imaginary part of impedance')
    plt.plot(t, Vodd)


# Desired Vector and desired response -----------------------------------------
Pfit = Lfilling + Ntap

Pfit_ = np.linspace(-(Pfit-1)/2, (Pfit-1)/2, Pfit)

# Even symmetric part of beam loading
Dvectoreven = V_real(Pfit_, Lfilling)

# Odd symmetric part of beam loading
Dvectorodd = V_imag(Pfit_, Lfilling)

# Total
Dvector = Dvectoreven + Dvectorodd

if PLT_DESIRED_VEC:
    plt.figure('Even symmetric part of beam loading')
    plt.title('Even symmetric part of beam loading')
    plt.plot(Dvectoreven, '.')

    plt.figure('Odd symmetric part of beam loading')
    plt.title('Odd symmetric part of beam loading')
    plt.plot(Dvectorodd, '.')

    plt.figure('Even symmetric part of beam loading')
    plt.title('Even symmetric part of beam loading')
    plt.plot(Dvectoreven, '.')

    plt.figure('Total of beam loading')
    plt.title('Total of beam loading')
    plt.plot(Dvector, '.')



# Optimal FIR by splitting real and imaginary parts of Zb ---------------------

# Weighting matrix
def uniform_weighting(n, Nt):
    return 1

def WeigthingUni(Nt):
    output = np.zeros((Nt, Nt))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if i == j:
                output[i,j] = uniform_weighting(i, Nt)
    return output


# Optimal FIR for real valued impedance (Even symmetric beam loading)
def EvenMatrix(Nt):
    output = np.zeros((Nt, (Nt + 1)//2))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if i + j == (Nt + 0)//2:
                output[i,j] = 1
            elif i - j == (Nt - 1)//2:
                output[i,j] = 1
    return output

def Hoptreal(Nt, L, P):
    output = np.matmul(np.transpose(Rmatrix(P, L)), Dvectoreven)
    output = np.matmul(np.transpose(Smatrix(P + L - 1, Nt)), output)
    output = np.matmul(np.transpose(EvenMatrix(Nt)), output)

    matrix1 = EvenMatrix(Nt)
    matrix1 = np.matmul(Smatrix(P + L - 1, Nt), matrix1)
    matrix1 = np.matmul(Rmatrix(P, L), matrix1)
    matrix1 = np.matmul(WeigthingUni(P), matrix1)
    matrix1 = np.matmul(np.transpose(Rmatrix(P, L)), matrix1)
    matrix1 = np.matmul(np.transpose(Smatrix(P + L - 1, Nt)), matrix1)
    matrix1 = np.matmul(np.transpose(EvenMatrix(Nt)), matrix1)
    matrix1 = npla.inv(matrix1)

    return np.matmul(matrix1, output)

def Hopteven(Nt, L, P):
    return np.concatenate([Hoptreal(Nt, L, P)[1:][::-1], Hoptreal(Nt, L, P)])

hopteven = Hopteven(Ntap, Lfilling, Pfit)


# Optimal FIR for imaginary valued impedance (Odd symmetric beam loading)
def OddMatrix(Nt):
    output = np.zeros((Nt, (Nt - 1) // 2))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if i + j == (Nt - 2) // 2:
                output[i, j] = -1
            elif i - j == (Nt + 1) // 2:
                output[i, j] = 1
    return output

def Hoptimag(Nt, L, P):
    output = np.matmul(WeigthingUni(P), Dvectorodd)
    output = np.matmul(np.transpose(Rmatrix(P, L)), output)
    output = np.matmul(np.transpose(Smatrix(P + L - 1, Nt)), output)
    output = np.matmul(np.transpose(OddMatrix(Nt)), output)

    matrix1 = OddMatrix(Nt)
    matrix1 = np.matmul(Smatrix(P + L - 1, Nt), matrix1)
    matrix1 = np.matmul(Rmatrix(P, L), matrix1)
    matrix1 = np.matmul(WeigthingUni(P), matrix1)
    matrix1 = np.matmul(np.transpose(Rmatrix(P, L)), matrix1)
    matrix1 = np.matmul(np.transpose(Smatrix(P + L - 1, Nt)), matrix1)
    matrix1 = np.matmul(np.transpose(OddMatrix(Nt)), matrix1)
    matrix1 = npla.inv(matrix1)

    return np.matmul(matrix1, output)

def Hoptodd(Nt, L, P):
    output = np.concatenate([-Hoptimag(Nt, L, P)[::-1], np.array([0])])
    return np.concatenate([output, Hoptimag(Nt, L, P)])

hoptodd = Hoptodd(Ntap, Lfilling, Pfit)

if PLT_OPT_FIR:
    plt.figure('Optimal FIR for real valued impedance')
    plt.title('Optimal FIR for real valued impedance')
    plt.plot(hopteven, '.')

    plt.figure('Optimal FIR for imaginary valued impedance')
    plt.title('Optimal FIR for imaginary valued impedance')
    plt.plot(hoptodd, '.')

    print('Error in real FIR:', np.mean(np.abs((hopteven - phopteven))))
    print('Error in imaginary FIR:', np.mean(np.abs((hoptodd - phoptodd))))


# Reconstructed signal. Step response of FIR + rectangular window (cavity) ----
def Yeven(Nt, L, P):
    output = np.matmul(Smatrix(P + L - 1, Nt), Hopteven(Nt, L, P))
    return np.matmul(Rmatrix(P, L), output)

def Yodd(Nt, L, P):
    output = np.matmul(Smatrix(P + L - 1, Nt), Hoptodd(Nt, L, P))
    return np.matmul(Rmatrix(P, L), output)

def Ytotal(Nt, L, P):
    return Yeven(Nt, L, P) + Yodd(Nt, L, P)

def Ytotal2(Nt, L, P):
    output = Hopteven(Nt, L, P) + Hoptodd(Nt, L, P)
    output = np.matmul(Smatrix(P + L - 1, Nt), output)
    return np.matmul(Rmatrix(P, L), output)

# TODO: Max error and std as a function of number of taps

if PLT_REC_SIG:
    plt.figure('Yeven')
    plt.title('Yeven')
    plt.plot(Yeven(Ntap, Lfilling, Pfit), '.')

    plt.figure('Yodd')
    plt.title('Yodd')
    plt.plot(Yodd(Ntap, Lfilling, Pfit), '.')

    plt.figure('Ytotal')
    plt.title('Ytotal')
    plt.plot(Ytotal(Ntap, Lfilling, Pfit), '.')
    plt.plot(Dvector, '.')

    plt.figure('Y Error')
    plt.title('Y Error')
    Yerror = Ytotal(Ntap, Lfilling, Pfit) - Dvector
    plt.plot(Yerror, '.')
    print('---- Y Error with splitting ----')
    print('Max error:', np.max(np.abs(Yerror)))
    print('Sigma error:', np.std(Yerror))

# Compare with FF coefficients from BLonD
bl_3sec = feedforward_filter_TWC3
bl_4sec = feedforward_filter_TWC4

my_3sec = Hopteven(Ntap, Lfilling, Pfit) + Hoptodd(Ntap, Lfilling, Pfit)

print(my_3sec)
plt.figure()
plt.plot(bl_4sec, '.')
plt.plot(my_3sec, '.')

# Optimal FIR without splitting even and odd ----------------------------------
def w(n, Nt):
    return 1 - 1 / 2 * np.hanning(Nt)[n]

def WeigthingHann(Nt):
    output = np.zeros((Nt, Nt))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if i == j:
                output[i,j] = w(i, Nt)
    return output

# Optimal FIR without splitting in even and odd
def Hopt(Nt, L, P):
    output = np.matmul(WeigthingHann(P), Dvector)
    output = np.matmul(np.transpose(Rmatrix(P, L)), output)
    output = np.matmul(np.transpose(Smatrix(P + L - 1, Nt)), output)

    matrix1 = np.matmul(Rmatrix(P, L), Smatrix(P + L - 1, Nt))
    matrix1 = np.matmul(WeigthingHann(P), matrix1)
    matrix1 = np.matmul(np.transpose(Rmatrix(P, L)), matrix1)
    matrix1 = np.matmul(np.transpose(Smatrix(P + L - 1, Nt)), matrix1)
    matrix1 = npla.inv(matrix1)
    return np.matmul(matrix1, output)

if PLT_OPT_FIR_WO:
    plt.matshow(WeigthingHann(Ntap))

    plt.figure('Hann')
    plt.title('Hann')
    s = np.linspace(0, Ntap-1, Ntap-1, dtype=int)
    plt.plot(w(s, Ntap), '.')

    plt.figure('Hopt')
    plt.title('Hopt')
    plt.plot(Hopt(Ntap, Lfilling, Pfit), '.')

    plt.figure('Hopt + reversed Hopt')
    plt.title('Hopt + reversed Hopt')
    plt.plot(0.5 * (Hopt(Ntap, Lfilling, Pfit) + Hopt(Ntap, Lfilling, Pfit)[::-1]), '.')

    plt.figure('Compare Hopt with splitting')
    plt.title('Compare Hopt with splitting')
    plt.plot(Hopt(Ntap, Lfilling, Pfit) - Hopteven(Ntap, Lfilling, Pfit), - Hoptodd(Ntap, Lfilling, Pfit), '.')

# Reconstructed signal from Hopt ----------------------------------------------
def YT(Nt, L, P):
    output = np.matmul(Smatrix(P + L - 1, Nt), Hopt(Nt, L, P))
    return np.matmul(Rmatrix(P, L), output)


if PLT_REC_SIG_WO:
    plt.figure('Reconstructed signal without splitting')
    plt.title('Reconstructed signal without splitting')
    plt.plot(YT(Ntap, Lfilling, Pfit), '.')

    plt.figure('Reconstructed signal with and without splitting')
    plt.title('Reconstructed signal with and without splitting')
    plt.plot(YT(Ntap, Lfilling, Pfit), '.')
    plt.plot(Ytotal(Ntap, Lfilling, Pfit), '.')

    plt.figure('Error from signal without splitting')
    plt.title('Error from signal without splitting')
    plt.plot(YT(Ntap, Lfilling, Pfit) - Dvector, '.')

    Yerror = YT(Ntap, Lfilling, Pfit) - Dvector
    print('---- Y Error without splitting ----')
    print('Max error:', np.max(np.abs(Yerror)))
    print('Sigma error:', np.std(Yerror))

plt.show()