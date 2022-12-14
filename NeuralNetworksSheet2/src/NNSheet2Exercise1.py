from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
import numpy as np
from math import *
from matplotlib import cm

# Constants
samerange = 2501
length = 1
c = 20.0
xrange = samerange
trange = samerange
dx = 0.0004
dt = 0.00002
xmax = xrange - 1
tmax = trange - 1
k = 4 * pi / length
multiplier = (c * c * dt * dt / (dx * dx))

# Time Steps
xsteps = np.array([dx * i for i in range(xrange)])
tsteps = np.array([dt * i for i in range(trange)])

# Function Psi
# calculated using list comprehensions as designed
# For explenation look up the manual calculations in
psi = np.array([np.array([0.0 for x in xsteps]) for t in tsteps])
psi[0] = np.array(
    [0 if (xn == 0 or xn == xrange - 1) else
     5 * sin(k * xsteps[xn]) for xn in range(xrange)])
psi[1] = np.array([0 if (xn == 0 or xn == xmax)
                   else 5 * sin(k * xsteps[xn]) + multiplier *
                        (2.5 * sin(k * xsteps[xn - 1]) +
                         2.5 * sin(k * xsteps[xn + 1]) -
                         5 * sin(k * xsteps[xn])) for xn in range(xrange)])
for tn in range(1, tmax):
    psi[tn + 1] = np.array([0 if (xn == 0 or xn == xmax)
                            else 2 * psi[tn, xn] - psi[tn - 1, xn] + multiplier * (
            psi[tn, xn - 1] + psi[tn, xn + 1] - 2 * psi[tn, xn]) for xn in range(xrange)])

# Setting the parameters for plotting via matplotlib
figconst1 = 111
figure = plt.figure()
axes = figure.add_subplot(figconst1, projection='3d')
(x, y) = np.meshgrid(xsteps, tsteps)
z = psi

# Create an empty array of strings with the same shape as the meshgrid, and
# populate it with two colors in a checkerboard pattern.
colortuple = ('yellow', 'blue')
colors = np.array(
    [np.array([colortuple[int((xn + yn) % len(colortuple))] for xn in range(xrange)]) for yn in range(trange)])

# Plotting function
colormap_index = 2

if colormap_index == 1:
    axes.plot_surface(x, y, z, facecolors=colors)
elif colormap_index == 2:
    axes.plot_surface(x, y, z, cmap=cm.coolwarm)

# Customize the z axis.
axes.zaxis.set_major_locator(LinearLocator(6))
axes.set_xlabel(r"$x\,[\mathrm{m}]$")
axes.set_ylabel(r"$t\,[\mathrm{s}]$")
axes.set_zlabel(r"$\psi\,(x,t)$")

plt.show()
