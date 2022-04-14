'''
SPS cable transfer function.

Author: Danilo Quartullo
'''
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

PLT = False

def cables_tranfer_function(profile_time, profile_current):
    
    # Import the CTF unfiltered
    dir_fil = os.path.dirname(os.path.abspath(__file__))
    data = np.load(dir_fil + '/' + 'cableTF.npz')
    tf = data['transfer']
    freq_tf = data['freqArray']
    Delta_f = freq_tf[1]-freq_tf[0] # 40 MHz
    t_max = 1/Delta_f
    f_max = freq_tf[-1]
    Delta_t = 1/(2*f_max) # 50 ps
    n_fft = 2*(len(tf)-1)
    
    # Apply raised cosine filter
    H_RC = np.zeros(len(freq_tf))
    cutoff_left = 2.5e9
    cutoff_right = 3.0e9
    index_inbetween = np.where((freq_tf<=cutoff_right)&(freq_tf>=cutoff_left))[0]
    index_before = np.where(freq_tf<cutoff_left)[0]
    index_after = np.where(freq_tf>cutoff_right)[0]
    H_RC[index_before] = 1
    H_RC[index_after] = 0
    H_RC[index_inbetween] = (1+np.cos(np.pi/(cutoff_right-cutoff_left)*(freq_tf[index_inbetween]-cutoff_left)))/2
    tf_RC = tf*H_RC
    
    # Interpolate if dt != 50 ps
    profile_dt = profile_time[1] - profile_time[0]
    if profile_dt != Delta_t:
        n_fft_new = max(len(profile_time), n_fft)
        if n_fft_new%2!=0:
            n_fft_new += 1
        freq_tf_new = np.fft.rfftfreq(n_fft_new, profile_dt)
        tf_real_new = np.interp(freq_tf_new, freq_tf, np.real(tf_RC))
        tf_imag_new = np.interp(freq_tf_new, freq_tf, np.imag(tf_RC))
        tf_new = tf_real_new + 1j * tf_imag_new
    else:
        tf_new = tf_RC
    
    # Apply the filtered CTF
    profile_spectrum = np.fft.rfft(profile_current, n=n_fft_new)
    profSpectrum_CTF = profile_spectrum*tf_new
    CTF_profile = np.fft.irfft(profSpectrum_CTF)[:len(profile_time)]
    
    return CTF_profile

if PLT:
    # Create profile
    bin_size = 0.05e-9
    profile_time = np.arange(-5*1e-9, 5*1e-9, bin_size)
    profile_current = norm.pdf(profile_time, 0, 1e-9)

    # Apply transfer function
    CTF_profile = cables_tranfer_function(profile_time, profile_current)

    # Plot
    ax = plt.gca()
    ax.plot(profile_time*1e9, profile_current, color='b')
    ax.plot(profile_time*1e9, CTF_profile*np.max(profile_current)/np.max(CTF_profile), color='r')
    ax.grid()
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Profile [a.u.]')
    plt.show()