from matplotlib import pyplot as plt
from AnalyticalNetwork import Network
from numpy import sin, cos, pi, mgrid, array

# Define training variables
training_size = 1000
training_range = range(training_size)
loops = 10
xy_limit = 2

# Training data
x_training = [xy_limit * xy_limit * (k / training_size) * sin(loops * pi * k / training_size)
              if k < (training_size / xy_limit) else
              -xy_limit * xy_limit * ((k - (training_size / xy_limit)) / training_size) * sin(
                  loops * pi * (k - (training_size / xy_limit)) / training_size)
              for k in training_range]  # noqa
y_training = [xy_limit * xy_limit * (k / training_size) * cos(loops * pi * k / training_size)
              if k < (training_size / xy_limit) else
              -xy_limit * xy_limit * ((k - (training_size / xy_limit)) / training_size) * cos(
                  loops * pi * (k - (training_size / xy_limit)) / training_size)
              for k in training_range]  # noqa
expected_output = [xy_limit if k < (training_size / xy_limit) else -xy_limit for k in training_range]
training_data = [[(x_training[k], y_training[k]), expected_output[k]] for k in training_range]


# Color function returns the color in dependence of the point's value
def color_function(xy_sign: float):
    return 'red' if xy_sign < 0.0 else 'blue'


colors = []
for i in training_range:
    colors.append(color_function(expected_output[i]))
for i in training_range:
    plt.scatter(x_training[i], y_training[i],
                c=colors[i],
                s=50,
                linewidths=0)
plt.show()
plt.close()

# Initialisation variables
nbins = 100
network = Network()

# initial network results
x, y = mgrid[-xy_limit:xy_limit:nbins * 1j, -xy_limit:xy_limit:nbins * 1j]
z = []
for xi, yi in zip(x.flatten(), y.flatten()):
    output = network.output(xi, yi)
    z.append(output)
z = array(z)
plt.pcolormesh(x, y, z.reshape(x.shape), shading='auto')
plt.show()
plt.close()

# Training
network.gradient_descent(training_data=training_data,
                         epochs=1001,
                         eta=0.02,
                         epoch_output=1)
# initial network results
x, y = mgrid[-xy_limit:xy_limit:nbins * 1j, -xy_limit:xy_limit:nbins * 1j]
z = []
for xi, yi in zip(x.flatten(), y.flatten()):
    output = network.output(xi, yi)
    z.append(output)
z = array(z)
plt.pcolormesh(x, y, z.reshape(x.shape), shading='auto')
plt.show()
plt.close()
