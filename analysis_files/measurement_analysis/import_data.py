import numpy as np
from scipy.stats import linregress



def measured_offset():
    pos_tot = np.load('../../data_files/beam_measurements/bunch_positions_total_red.npy')
    pos_fl = pos_tot.reshape((25 * 100, 288))
    pos_fb = pos_fl[:,:72]
    b_n = np.linspace(1, 72, 72)
    pds = np.zeros(pos_fb.shape)

    for i in range(pos_fb.shape[0]):
        s1, i1, rval, pval, stderr = linregress(b_n, pos_fb[i,:])

        pds[i,:] = pos_fb[i,:] - s1 * b_n - i1

    avg_pd = np.mean(pds, axis = 0)
    std_pd = np.std(pds, axis = 0)

    return avg_pd, std_pd



def get_power():
    dt = 8e-9
    t = np.linspace(0, dt * 65536, 65536)
    sec3_data = np.zeros((65536, 4, 3))
    sec4_data = np.zeros((65536, 2, 3))

    for i in range(6):
        for j in range(3):
            if i == 0:
                sec3_data[:, 0, j] = np.load(f'../../data_files/power_measurements/power_cav{i + 1}_meas{j}.npy')
            elif i == 1:
                sec3_data[:, 1, j] = np.load(f'../../data_files/power_measurements/power_cav{i + 1}_meas{j}.npy')
            elif i == 2:
                sec4_data[:, 0, j] = np.load(f'../../data_files/power_measurements/power_cav{i + 1}_meas{j}.npy')
            elif i == 3:
                sec3_data[:, 2, j] = np.load(f'../../data_files/power_measurements/power_cav{i + 1}_meas{j}.npy')
            elif i == 4:
                sec3_data[:, 3, j] = np.load(f'../../data_files/power_measurements/power_cav{i + 1}_meas{j}.npy')
            elif i == 5:
                sec4_data[:, 1, j] = np.load(f'../../data_files/power_measurements/power_cav{i + 1}_meas{j}.npy')

    sec3_mean = np.mean(sec3_data, axis=2)
    sec3_mean_tot = np.mean(sec3_mean, axis=1)

    sec3_std = np.mean(sec3_data, axis=2)
    sec3_std_tot = np.std(sec3_std, axis=1)

    sec4_mean = np.mean(sec4_data, axis=2)
    sec4_mean_tot = np.mean(sec4_mean, axis=1)

    sec4_std = np.mean(sec4_data, axis=2)
    sec4_std_tot = np.std(sec4_std, axis=1)

    return sec3_mean_tot, sec3_std_tot, sec4_mean_tot, sec4_std_tot
