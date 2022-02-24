'''
This is a test file to understand and make simple calculations in preparation for the SPS FF model

Author: Birk Emil Karlsen-Bæck
'''

PLT_IMPULSE = False
PLT_BEAM_LOADING = False
PLT_DESIRED_VEC = False
PLT_OPT_FIR = True

# Imports ---------------------------------------------------------------------
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
import utility_files.analysis_tools as at

plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fourier}',
        'font.family': 'serif',
        'font.size': 16
    })


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
T = 420e-9                              # Filling time [s]
Lfilling = round(T * fs)                # Cavity response length
Ntap = 31                               # Filter response length. MUST BE ODD


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

def Weigthing(Nt):
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
    matrix1 = np.matmul(Weigthing(P), matrix1)
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
    output = np.matmul(Weigthing(P), Dvectorodd)
    output = np.matmul(np.transpose(Rmatrix(P, L)), output)
    output = np.matmul(np.transpose(Smatrix(P + L - 1, Nt)), output)
    output = np.matmul(np.transpose(OddMatrix(Nt)), output)

    matrix1 = OddMatrix(Nt)
    matrix1 = np.matmul(Smatrix(P + L - 1, Nt), matrix1)
    matrix1 = np.matmul(Rmatrix(P, L), matrix1)
    matrix1 = np.matmul(Weigthing(P), matrix1)
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

















plt.show()