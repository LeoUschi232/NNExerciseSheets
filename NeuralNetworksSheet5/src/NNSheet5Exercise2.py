from numpy import pi, linspace, random, array, sin, set_printoptions, linalg
from matplotlib import pyplot as plt, use
from sympy import Symbol, lambdify

set_printoptions(suppress=True, precision=4)
use('module://backend_interagg')
print("NN Sheet 5 Exercise 2 process started\n")

# Function that will be worked with
k = 2 * pi


def f1(x):
    return sin(k * x)


# Uniformly generated x-values in interval [0, 1]
xsize = 25
xmin = 0.0
xmax = 1.0
precision = 200
xvalues = linspace(xmin, xmax, xsize)
xpoints = linspace(0, xvalues[len(xvalues) - 1], precision)

# y-values with pseudo-random errors
error = 0.75
yvalues = array([f1(x) + random.uniform(-error, error) for x in xvalues])

polyorders = range(1, 10)
for polyorder in polyorders:
    # Defining the poly-fit function
    poly_range = range(polyorder)
    a = array([Symbol(f'a{i}') for i in range(polyorder)])

    # Chi-squared-function as necessary
    correction = 1.0e-6
    chi = sum([(yvalues[j] - sum([a[i] * xvalues[j] ** i for i in range(polyorder)])) ** 2
               for j in range(len(xvalues))]) \
          + correction * sum([a[i] * a[i] for i in range(polyorder)])

    # Multi-dimensional vectors and matrices to minimize error
    raw_beta = [chi.diff(a[i]) for i in poly_range]
    beta = lambdify([a[i] for i in poly_range], raw_beta, 'numpy')
    raw_alpha = [[raw_beta[i].diff(a[j]) for j in poly_range] for i in poly_range]
    alpha = lambdify([a[i] for i in poly_range], raw_alpha, 'numpy')


    # Utilization of the inverse alpha matrix
    def alpha_inverse(p_avalues):
        return array(linalg.inv(array(alpha(*array([ai for ai in p_avalues])))))


    # Calculating the minimal a values
    max_depth = 10
    first_value = 0.0
    adata = array([array([0.0 for j in poly_range]) for i in range(max_depth)])
    adata[0] = array([first_value for j in poly_range])


    # Polynomial function to approx function
    def y1(args, x):
        return sum([args[i] * x ** i for i in poly_range])


    # Recursively minimizing error
    for i in range(1, max_depth):
        avalues = adata[i - 1]
        beta_array = beta(*array([ai for ai in avalues]))
        alphainv_array = array(alpha_inverse(avalues))
        adata[i] = array(avalues - alphainv_array @ beta_array)

    arguments = array(adata[max_depth - 1])

    # Define final function
    fitstyle = '-r'
    plt.plot(xpoints, y1(arguments, xpoints), fitstyle)

# Plotting the values against each other
values_style = '.b'
function_style = '-b'
plt.plot(xvalues, yvalues, values_style, xpoints, f1(xpoints), function_style)
plt.show()

print("NN Sheet 5 Exercise 2 process finished\n")
