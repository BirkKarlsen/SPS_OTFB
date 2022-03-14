'''
This file was made to us the bisection method to optimize the transmitter gain by calling the other file.

Note: optimization of the transmitter gain usually takes about 45-50 iterations. The transmitter gain for the
3-section should be between 0.2 and 0.3 and the 4-section should be between 0.4 and 0.7. Unless the central
frequency is such that the feedback is at the edge of stability, in which case the transmitter gain is
usually between 0.1 and 0.2 for both cavities.

Author: Birk Emil Karlsen-Bæck
'''


# Imports ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import subprocess


# Get result from optimization script -----------------------------------------
def get_output(g, G3SEC, gs):
    if G3SEC:
        proc = subprocess.Popen(['python', 'optimize_transmittergains.py', '-bs', '1', '-g1', f'{g}', '-g2', f'{gs}'],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.Popen(['python', 'optimize_transmittergains.py', '-bs', '1', '-g1', f'{gs}', '-g2', f'{g}'],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    output = np.array(proc.communicate()[0].split()).astype(float)
    if G3SEC:
        return output[0]
    else:
        return output[1]


# Bisection Method ------------------------------------------------------------
def bisection_method(target, a, b, G3SEC, gs, max_it, tol):
    ya = get_output(a, G3SEC, gs)
    yb = get_output(b, G3SEC, gs)

    c = (a + b)/2
    yc = get_output(c, G3SEC, gs)

    err = (target - yc) / target * 100

    i = 0
    while abs(err) > tol:
        i += 1
        if i > max_it:
            print("Failed to converge")
            return c, yc

        if (target - yc) * (target - ya) < 0:
            b = c
            yb = yc
        else:
            a = c
            ya = yc

        c = (a + b)/2
        yc = get_output(c, G3SEC, gs)
        err = (target - yc) / target * 100
        print(f"Iteration {i}, error is {err}%")

    return c, yc

# Lauch optimization ----------------------------------------------------------
V_part = 0.60
V = 10
# 3-section
target = V * V_part
a = 0.9
b = 1.1

c3, yc3 = bisection_method(target, a, b, True, 0.4, 100, 1e-14)

print('----- 3-section -----')
print(c3, yc3)

# 4-section
target = V * (1 - V_part)
a = 0.9
b = 1.1

c4, yc4 = bisection_method(target, a, b, False, 0.4, 100, 1e-14)

print('----- 3-section -----')
print(c3, yc3)
print('----- 4-section -----')
print(c4, yc4)
