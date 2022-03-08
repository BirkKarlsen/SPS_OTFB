'''
This file was made to us the bisection method to optimize the transmitter gain by calling the other file

Author: Birk Emil Karlsen-Bæck
'''


# Imports ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import subprocess


proc = subprocess.Popen(['python', 'optimize_transmittergains.py', '-bs', '1'], stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
print(proc.communicate())

# Bisection Method ------------------------------------------------------------
