from sys import exit
import matplotlib.pyplot as plt
import numpy as np
from math import *
from matplotlib import cm
import pandas as pd

# Constants and Variables
length = 20
c = 20.0
xymax = 200
xmax = xymax
ymax = xymax
tmax = 500
xrange = range(xmax + 1)
yrange = range(ymax + 1)
trange = range(tmax + 1)
dxy = 0.1
dx = dxy
dy = dxy
dt = 0.002
xyoffset = -10
xoffset = xyoffset
yoffset = xyoffset
kexp = -0.02
ksin = pi / 5
multiplier = c * c * dt * dt
assert c * dt <= dx and c * dt <= dy

print(multiplier / (dx * dx))
print(multiplier / (dy * dy))

# Steps
xsteps = np.array([xoffset + dx * i for i in xrange])
ysteps = np.array([yoffset + dy * i for i in xrange])
tsteps = np.array([dt * i for i in trange])

# Function Psi
# calculated using list comprehensions as designed
# For explenation look up the manual calculations in
psi = np.array([np.array([np.array([0.0 for y in ysteps]) for x in xsteps]) for t in tsteps])

psi[0] = np.array([np.array([exp(kexp * (x * x + y * y)) * sin(ksin * x) * sin(ksin * y)
                             for y in ysteps]) for x in xsteps])
psi0 = psi[0]
psi[1] = np.array([np.array([0.0 if (xn <= 1 or yn <= 1 or xn >= xmax - 1 or yn >= ymax - 1)
                             else psi0[yn, xn] + multiplier * (
        (psi0[yn, xn + 2] + psi0[yn, xn - 2] - 2 * psi0[yn, xn]) / (2 * dx * dx) +
        (psi0[yn + 1, xn + 1] + psi0[yn - 1, xn - 1] - psi0[yn + 1, xn - 1] - psi0[yn - 1, xn + 1]) / (dx * dy) +
        (psi0[yn + 2, xn] + psi0[yn - 2, xn] - 2 * psi0[yn, xn]) / (2 * dy * dy))
                             for yn in yrange]) for xn in xrange])
"""
for yn in yrange:
    for xn in xrange:
        if fabs(psi[1, yn, xn]) > fabs(psi[0, yn, xn]):
            print(f"yn = {yn} and xn = {xn}\n{fabs(psi[1, yn, xn])} > {fabs(psi[0, yn, xn])}")
            exit(0)
"""

for tn in range(1, tmax):
    psi[tn + 1] = np.array([np.array([0.0 if (xn <= 1 or yn <= 1 or xn >= xmax - 1 or yn >= ymax - 1)
                                      else multiplier * (
            (psi[tn, yn, xn + 2] + psi[tn, yn, xn - 2] - 2 * psi[tn, yn, xn]) / (dx * dx) +
            2 * (psi[tn, yn + 1, xn + 1] + psi[tn, yn - 1, xn - 1] -
                 psi[tn, yn + 1, xn - 1] - psi[tn, yn - 1, xn + 1]) / (dx * dy) +
            (psi[tn, yn + 2, xn] + psi[tn, yn - 2, xn] - 2 * psi[tn, yn, xn]) / (dy * dy))
                                           - psi[tn - 1, yn, xn] + 2 * psi[tn, yn, xn]
                                      for yn in yrange]) for xn in xrange])

# Saving psi function
for tn in trange:
    pd.DataFrame(psi[tn]).to_csv(f"PsiValues/psi_t{tn}.csv")

# constants for plotting via matplotlib
figconst = 111
dz = 50
zrange = 11

# parameters for plotting via matplotlib
figures = np.array([plt.figure() for zn in range(zrange)])
axeses = np.array([figure.add_subplot(figconst, projection='3d')
                   for figure in figures])
(x, y) = np.meshgrid(xsteps, ysteps)
zlist = np.array([psi[dz * zn] for zn in range(zrange)])

# Calculate all plots
for zn in range(zrange):
    figure = figures[zn]
    axes = axeses[zn]
    axes.set_xlabel(r"$x\,[\mathrm{m}]$")
    axes.set_ylabel(r"$y\,[\mathrm{m}]$")
    axes.set_zlabel(r"$\psi\,(x,y,t={}s)$".format(tsteps[zn * dz]))
    z = psi[zn * dz]
    axes.plot_surface(x, y, z, cmap=cm.coolwarm)

# Display plot
plt.show()
