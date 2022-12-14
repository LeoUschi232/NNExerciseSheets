# 2. χ fit of a non-linear function (Breit-Wigner)
# In this exercise we are going to fit a non-linear function.
# The dataset
# double datax[9] = {0,25,50,75,100,125,150,175,200};
# double datay[9] = {10.6,16.0,45.0,83.5,52.8,19.9,10.8,8.25,4.7};
# double datasigma[9] ={2.3, 3.5, 4.5, 6.4, 4.4, 3.4, 2.1, 1.6, 1.1};
# can be copied and pasted from moodle.

from numpy import array, linalg, linspace, set_printoptions
from matplotlib import pyplot as plt
from sympy import Symbol, lambdify

set_printoptions(suppress=True, precision=4)

# Defining data sets
datax = [0, 25, 50, 75, 100, 125, 150, 175, 200]
datay = [10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7]
datasigma = [2.3, 3.5, 4.5, 6.4, 4.4, 3.4, 2.1, 1.6, 1.1]

# Defining the fit function
a1 = Symbol('a1')
a2 = Symbol('a2')
a3 = Symbol('a3')
chi = sum([((datay[i] - (a1 / ((a2 - datax[i]) ** 2 + a3))) / datasigma[i]) ** 2 for i in range(len(datax))])

# Lambdifying the beta-function
k = 0.5
beta1 = -k * chi.diff(a1)
beta2 = -k * chi.diff(a2)
beta3 = -k * chi.diff(a3)
beta = lambdify([a1, a2, a3], [beta1, beta2, beta3], 'numpy')

# Lambdifying the alpha function
alpha11 = k * chi.diff(a1).diff(a1)
alpha12 = k * chi.diff(a1).diff(a2)
alpha13 = k * chi.diff(a1).diff(a3)
alpha21 = k * chi.diff(a2).diff(a1)
alpha22 = k * chi.diff(a2).diff(a2)
alpha23 = k * chi.diff(a2).diff(a3)
alpha31 = k * chi.diff(a3).diff(a1)
alpha32 = k * chi.diff(a3).diff(a2)
alpha33 = k * chi.diff(a3).diff(a3)
alpha = lambdify([a1, a2, a3], [[alpha11, alpha21, alpha31],
                                [alpha12, alpha22, alpha32],
                                [alpha13, alpha23, alpha33]], 'numpy')


def alpha_inverse(aa1, aa2, aa3):
    return array(linalg.inv(alpha(aa1, aa2, aa3)))


# Calculating the minimal chi values
dimension = 3
max_depth = 10
adata = array([array([0.0 for j in range(dimension)]) for i in range(max_depth)])
adata[0] = array([60000.0, 70, 550])

for i in range(1, max_depth):
    print("Current depth:", i)
    a = adata[i - 1]
    print(f"adata[{i - 1}]", a)
    beta_array = array(beta(a[0], a[1], a[2]))
    alphainv_array = array(alpha_inverse(a[0], a[1], a[2]))
    print("Beta:", beta_array)
    adata[i] = array(a + alphainv_array @ beta_array)
    print(f"adata[{i}]", adata[i], "\n")

# Define final function
argument1 = adata[max_depth - 1][0]
argument2 = adata[max_depth - 1][1]
argument3 = adata[max_depth - 1][2]

print("arguments: ", argument1, argument2, argument3)


def y(x):
    return argument1 / ((argument2 - x) ** 2 + argument3)


# Plot all datapoints together with the function
precision = 200
xpoints = linspace(0, datax[len(datax) - 1], precision)
plt.plot(datax, datay, 'ro', xpoints, y(xpoints))
plt.show()
