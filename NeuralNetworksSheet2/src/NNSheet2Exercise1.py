from matplotlib.ticker import LinearLocator
from matplotlib import pyplot as plt
from numpy import pi, sin, array, meshgrid

# Constants
xmax = 150
xrange = range(xmax + 1)
c = 20.0
dx = 1 / xmax
dt = 0.05 / xmax
k = 4 * pi
multiplier = (c * c * dt * dt / (dx * dx))

# Time Steps
xsteps = array([dx * i for i in xrange])
tsteps = array([dt * i for i in xrange])

# Function Psi
# calculated using list comprehensions as designed
# For explenation look up the manual calculations
psi = array([array([0.0 for x in xsteps]) for t in tsteps])
psi[0] = array([
    0 if (xn == 0 or xn == xmax) else
    5 * sin(k * xsteps[xn]) for xn in xrange
])
psi[1] = array([
    0 if (xn == 0 or xn == xmax) else
    5 * sin(k * xsteps[xn]) + multiplier
    * (2.5 * sin(k * xsteps[xn - 1]) + 2.5 * sin(k * xsteps[xn + 1]) - 5 * sin(k * xsteps[xn]))
    for xn in xrange
])
for tn in range(1, xmax):
    psi[tn + 1] = array([
        0 if (xn == 0 or xn == xmax)
        else 2 * psi[tn, xn] - psi[tn - 1, xn] + multiplier * (psi[tn, xn - 1] + psi[tn, xn + 1] - 2 * psi[tn, xn])
        for xn in xrange
    ])

# Setting the parameters for plotting via matplotlib
axes = plt.figure().add_subplot(projection="3d")
(x, y) = meshgrid(xsteps, tsteps)
z = psi

# Create an empty array of strings with the same shape as the meshgrid, and
# populate it with two colors in a checkerboard pattern.
colortuple = ("red", "blue")
colors = array([
    array([
        colortuple[int((xn + yn) % len(colortuple))] for xn in xrange
    ]) for yn in xrange
])

# Plotting function.
axes.plot_surface(x, y, z, facecolors=colors, rcount=len(xsteps), ccount=len(tsteps))

# Customize the z axis.
axes.zaxis.set_major_locator(LinearLocator(6))
axes.set_xlabel(r"$x\,[\mathrm{m}]$")
axes.set_ylabel(r"$t\,[\mathrm{s}]$")
axes.set_zlabel(r"$\psi\,(x,t)$")
plt.tight_layout()
plt.show()
