'''
File to check that the imported impedance objects are properly imported into the simulations.

Author: Birk Emil Karlsen-Bæck
'''

# Imports -------------------------------------------------------------------------------------------------------------
print("Importing...\n")
import numpy as np
import matplotlib.pyplot as plt

from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from blond.impedances.impedance_sources import InputTable

from SPS.impedance_scenario import scenario, impedance2blond

# Setting up needed objects for BLonD InducedVoltageFreq --------------------------------------------------------------
print("Setting up...\n")
ring = Ring(2 * np.pi * 1100.009, 1 / (18**2), 440e9, Proton(), n_turns=1)
rfstation = RFStation(ring, [4620], [6.7e6], [0])
beam = Beam(ring, int(5e5), int(1.15e11))
profile = Profile(beam, CutOptions=CutOptions(cut_left=rfstation.t_rf[0,0] * (1000 - 2.5),
        cut_right=rfstation.t_rf[0,0] * (1000 + 72 * 5 * 4 + 250 * 3 + 125),
        n_slices=int(round(2**7 * (2.5 + 72 * 5 * 4 + 250 * 3 + 125)))))

# The two scenarios that I want to compare ----------------------------------------------------------------------------
freqRes = 43.3e3

sc_str_1 = "200TWC_only.txt"
sc_str_2 = "200TWC_only_reduced.txt"
print("Importing Impedances...\n")
impScenario1 = scenario(sc_str_1)
impScenario2 = scenario(sc_str_2)

impModel1 = impedance2blond(impScenario1.table_impedance)
impModel2 = impedance2blond(impScenario2.table_impedance)

impFreq1 = InducedVoltageFreq(beam, profile, impModel1.impedanceList, freqRes)
impFreq2 = InducedVoltageFreq(beam, profile, impModel2.impedanceList, freqRes)

# Plotting the two scenarios ------------------------------------------------------------------------------------------
print("Plotting...\n")
plt.figure()
plt.plot(impFreq1.freq, impFreq1.total_impedance)
plt.plot(impFreq2.freq, impFreq2.total_impedance)

plt.show()
