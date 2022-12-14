# Root finding
# We want to write a program to find the roots of
# x^2 − 7x + 10
# using the Newton-Raphson method.
# To find out where to start looking for a root,
# you would usually have to either plot the functions or implement
# e.g. the bisection method to find an appropriate starting point,
# but here you should be able to do without this.
# Which precision (in terms of ϵm) can you expect?
# Make sure that your program terminates before reaching the point where additional steps worsen the result.
# What happens when you try to find the root of (the standard branch of) the arc
# tan function using x0 = 2 as the initial guess? Why?

from math import atan
from numpy import array


# Defining functions

def f1_0(x):
    return float(x * x - 7 * x + 10)


def f1_1(x):
    return float(2 * x - 7)


def f2_0(x):
    return float(atan(x))


def f2_1(x):
    return float(1 / (x * x + 1))


depth = 100

# Calculating the roots of f1(x)
# starting point x[0]
xarray = array([float("NaN") for i in range(depth)])
xarray[0] = 0.0
for i in range(1, depth):
    x1 = xarray[i - 1]
    xarray[i] = x1 - f1_0(x1) / f1_1(x1)
print("Numerical approximation for root #1 of f1(x):", xarray[depth - 1])
xarray = array([float("NaN") for i in range(depth)])
xarray[0] = 10.0
for i in range(1, depth):
    x1 = xarray[i - 1]
    xarray[i] = x1 - f1_0(x1) / f1_1(x1)
print("Numerical approximation for root #2 of f1(x):", xarray[depth - 1])

# Calculating the roots of f2(x)
# starting point x[0]
xarray = array([float("NaN") for i in range(depth)])
xarray[0] = 0.5
for i in range(1, depth):
    x1 = xarray[i - 1]
    xarray[i] = x1 - f2_0(x1) / f2_1(x1)
print("Numerical approximation for root of f2(x):", xarray[depth - 1])
xarray = array([float("NaN") for i in range(depth)])
xarray[0] = 2.0
for i in range(1, depth):
    x1 = xarray[i - 1]
    if f2_1(x1) == 0:
        break
    xarray[i] = x1 - f2_0(x1) / f2_1(x1)
print("Numerical approximation for root of f2(x):", xarray[depth - 1])
