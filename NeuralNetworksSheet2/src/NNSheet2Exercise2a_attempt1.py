from sys import exit
import matplotlib.pyplot as plt
import numpy as np
from math import *
from matplotlib import cm
import pandas as pd

# Constants and Variables
c = 20.0
xymin = -10
xymax = 10
xylength = xymax - xymin
xysteps = 200
xyrange = range(xysteps)
dxy = (xymax - xymin) / (xysteps - 1)
xyvalues = np.array(np.linspace(xymin, xymax, xysteps))
tsteps = 500
dt = dxy / (2 * c)
tmin = 0
tmax = (tsteps - 1) * dt
tlength = tmax - tmin
trange = range(tsteps)
tvalues = np.array(np.linspace(tmin, tmax, tsteps))
kexp = -0.02
ksin = pi / 5
multiplier = c * c * dt * dt / (dxy * dxy)
damping = 7.5e-14
assert c * dt <= dxy

# Function Psi
# calculated using list comprehensions as designed
# For explenation look up the manual calculations in
psi = np.array([np.array([np.array([0.0 for y in xyvalues]) for x in xyvalues]) for t in tvalues])

psi[0] = np.array([np.array([0.0 if (xn <= 0 or yn <= 0 or xn >= xysteps - 1 or yn >= xysteps - 1)
                             else exp(kexp * (xyvalues[xn] * xyvalues[xn] + xyvalues[yn] * xyvalues[yn]))
                                  * sin(ksin * xyvalues[xn]) * sin(ksin * xyvalues[yn])
                             for yn in xyrange]) for xn in xyrange])
psi0 = psi[0]
psi[1] = np.array([np.array([0.0 if (xn <= 0 or yn <= 0 or xn >= xysteps - 1 or yn >= xysteps - 1)
                             else (psi0[yn, xn] + 0.5 * multiplier * (
        (psi0[yn, xn + 1] + psi0[yn, xn - 1] - 2 * psi0[yn, xn]) +
        (psi0[yn + 1, xn] + psi0[yn - 1, xn] - 2 * psi0[yn, xn])))
                             for yn in xyrange]) for xn in xyrange])

for tn in range(1, tsteps - 1):
    psi[tn + 1] = np.array([np.array([0.0 if (xn <= 0 or yn <= 0 or xn >= xysteps - 1 or yn >= xysteps - 1)
                                      else (multiplier * (
            (psi[tn, yn, xn + 1] + psi[tn, yn, xn - 1] - 2 * psi[tn, yn, xn]) +
            (psi[tn, yn + 1, xn] + psi[tn, yn - 1, xn] - 2 * psi[tn, yn, xn]))
                                            - psi[tn - 1, yn, xn] + 2 * psi[tn, yn, xn])
                                      for yn in xyrange]) for xn in xyrange])

# constants for plotting via matplotlib
figconst = 111
dz = 50
zmax = 10
zrange = range(zmax)

# parameters for plotting via matplotlib
figures = np.array([plt.figure() for zn in zrange])
axeses = np.array([figure.add_subplot(figconst, projection='3d')
                   for figure in figures])
(x, y) = np.meshgrid(xyvalues, xyvalues)
zlist = np.array([psi[dz * zn] for zn in zrange])

# Calculate all plots
for zn in zrange:
    figure = figures[zn]
    axes = axeses[zn]
    axes.set_xlabel(r"$x\,[\mathrm{m}]$")
    axes.set_ylabel(r"$y\,[\mathrm{m}]$")
    axes.set_zlabel(r"$\psi\,(x,y,t={}s)$".format(tvalues[zn * dz]))
    z = psi[zn * dz]
    axes.plot_surface(x, y, z, cmap=cm.coolwarm)

# Display plot
plt.show()
