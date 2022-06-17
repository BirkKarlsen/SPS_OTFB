'''


Author: Birk Emil Karlsen-Bæck
'''


# Imports
import numpy as np
import matplotlib.pyplot as plt
import utility_files.analysis_tools as at
import json


# Setting
file_date = 20211106
file_timestamp = 103331
cavity_number = 1
file_dir = f'../../data_files/OTFB_november_measurements/' \
           f'sps_otfb_data__all_buffers__cavity{cavity_number}__flattop__{file_date}_{file_timestamp}.json'


f = open(file_dir)
data = json.load(f)
print(data.keys())

sampling_rate = data['SA.TWC200_expertVcavAmp.C1-ACQ']['samplingRate'] * 1e6
dt = 1 / sampling_rate
N_samples = len(data['SA.TWC200_expertVcavAmp.C1-ACQ']['data'])

t_array = np.linspace(0, dt * N_samples, N_samples)


antenna_voltage = np.array(data['SA.TWC200_expertVcavAmp.C1-ACQ']['data'])
antenna_phase = np.pi * np.array(data['SA.TWC200_expertVcavPhase.C1-ACQ']['data']) / 180
print(data['SA.TWC200_expertVcavAmp.C1-ACQ'].keys())
antenna_IQ = antenna_voltage * np.cos(antenna_phase) + 1j * antenna_voltage * np.sin(antenna_phase)

plt.figure()
plt.title('Antenna Voltage')
plt.plot(t_array, antenna_IQ.real, color='r')
plt.plot(t_array, antenna_IQ.imag, color='b')
plt.xlim((0, 2e-5))



OTFB_voltage = np.array(data['SA.TWC200_expertOTFBAmp.C1-ACQ']['data'])
OTFB_phase = np.pi * np.array(data['SA.TWC200_expertOTFBPhase.C1-ACQ']['data']) / 180
print(data['SA.TWC200_expertOTFBAmp.C1-ACQ'].keys())
print(data['SA.TWC200_expertOTFBAmp.C1-ACQ']['data_units'])
OTFB_IQ = OTFB_voltage * np.cos(OTFB_phase) + 1j * OTFB_voltage * np.sin(OTFB_phase)

plt.figure()
plt.title('OTFB Voltage')
plt.plot(t_array, OTFB_IQ.real, color='r')
plt.plot(t_array, OTFB_IQ.imag, color='b')
plt.xlim((0, 2e-5))

plt.figure()
plt.title('OTFB Phase')
plt.plot(t_array, (OTFB_phase + np.pi / 2) % (np.pi * 2))
plt.xlim((0, 2e-5))



RFdrive_voltage = np.array(data['SA.TWC200_expertRfDriveAmp.C1-ACQ']['data'])
RFdrive_phase = np.pi * np.array(data['SA.TWC200_expertRfDrivePhase.C1-ACQ']['data']) / 180
print(data['SA.TWC200_expertRfDriveAmp.C1-ACQ'].keys())
print(data['SA.TWC200_expertRfDriveAmp.C1-ACQ']['data_units'])
RFdrive_IQ = RFdrive_voltage * np.cos(RFdrive_phase) + 1j * RFdrive_voltage * np.sin(RFdrive_phase)

plt.figure()
plt.title('RF Drive Voltage')
plt.plot(t_array, RFdrive_IQ.real, color='r')
plt.plot(t_array, RFdrive_IQ.imag, color='b')
plt.xlim((0, 2e-5))

plt.figure()
plt.title('RF Drive Phase')
plt.plot(t_array, (RFdrive_phase + np.pi / 2) % (np.pi * 2))
plt.xlim((0, 2e-5))


'''
phase loop gain 150
otfb_static
'''




plt.show()