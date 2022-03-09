'''
This file was made to us the bisection method to optimize the transmitter gain by calling the other file

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
V_part = 0.5517843967841601
V = 6660589.53641675e-6
# 3-section
target = V * V_part
a = 0.2
b = 0.3

c, yc = bisection_method(target, a, b, True, 0.4, 100, 1e-14)

print(c, yc)
