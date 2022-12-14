# Try Levenberg-Marquardt for less well-informed starting values.
# Instead of the NR-step from the previous section, use

from numpy import array, linalg, linspace, set_printoptions
from matplotlib import pyplot as plt
from sympy import Symbol, lambdify

set_printoptions(suppress=True, precision=4)

# Defining data sets
datax = [0, 25, 50, 75, 100, 125, 150, 175, 200]
datay = [10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7]
datasigma = [2.3, 3.5, 4.5, 6.4, 4.4, 3.4, 2.1, 1.6, 1.1]
dimension = 3

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
alpha_main = lambdify([a1, a2, a3], [[alpha11, alpha21, alpha31],
                                     [alpha12, alpha22, alpha32],
                                     [alpha13, alpha23, alpha33]], 'numpy')
alpha_alt = lambdify([a1, a2, a3], [[alpha11, 0, 0],
                                    [0, alpha22, 0],
                                    [0, 0, alpha33]], 'numpy')

unitmatrix = array([array([1 if i == j else 0 for i in range(dimension)]) for j in range(dimension)])


def alpha_inverse(aa1, aa2, aa3):
    return array(linalg.inv(alpha_main(aa1, aa2, aa3)))


def inverse_type1(aa1, aa2, aa3, ll):
    return array(linalg.inv(array(alpha_main(aa1, aa2, aa3)) + ll * unitmatrix))


# Calculating the minimal chi values
l = 1
increase = 1
max_depth = 20
adata = array([array([0.0 for j in range(dimension)]) for i in range(max_depth)])
adata[0] = array([60000.0, 70, 550])
a = adata[0]
bestbeta = array(beta(a[0], a[1], a[2]))

for i in range(1, max_depth):
    a = adata[i - 1]
    beta_array = array(beta(a[0], a[1], a[2]))
    inv_array = array(inverse_type1(a[0], a[1], a[2], l))
    adata_i = array(a + inv_array @ beta_array)
    beta_arrayi = array(beta(adata_i[0], adata_i[1], adata_i[2]))
    if bestbeta @ bestbeta < beta_arrayi @ beta_arrayi:
        l += increase
        adata[i] = a
    else:
        l -= increase
        adata[i] = adata_i
        bestbeta = beta_arrayi
    print("Current depth:", i)
    print("l:", l)
    print("Best beta:", bestbeta)
    print("Beta array:", beta_arrayi)
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
