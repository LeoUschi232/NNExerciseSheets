from matplotlib import pyplot as plt
from NielsenNetwork import Network
from numpy import sin, cos, pi, mgrid, vstack, array

# Define instance variables
debug = 1
training_size = 1000
training_range = range(training_size)
loops = 10
test_size = 1600
test_range = range(test_size)
xy_limit = 2
size = 50
plot_training = 0
output_weight = 2
nbins = 100

# Training data
x_training = [2 * xy_limit * (k / training_size) * sin(loops * pi * k / training_size)
              if k < (training_size / 2) else
              -2 * xy_limit * ((k - (training_size / 2)) / training_size) * sin(
                  loops * pi * (k - (training_size / 2)) / training_size)
              for k in training_range]  # noqa
y_training = [2 * xy_limit * (k / training_size) * cos(loops * pi * k / training_size)
              if k < (training_size / 2) else
              -2 * xy_limit * ((k - (training_size / 2)) / training_size) * cos(
                  loops * pi * (k - (training_size / 2)) / training_size)
              for k in training_range]  # noqa
expected_output = [output_weight if k < (training_size / 2) else -output_weight for k in training_range]
training_data = [[(x_training[k], y_training[k]), expected_output[k]] for k in training_range]


# Color function returns the color in dependence of the supposed quadrant
# for positive quadrants (1 = top-right, 3 = bottom-left) -> blue
# for negative quadrants (2 = top-left, 4 = bottom-right) -> yellow
def color_function(xy_sign: float):
    return 'red' if xy_sign < 0.0 else 'blue'


colors = []
for i in training_range:
    colors.append(color_function(expected_output[i]))
for i in training_range:
    plt.scatter(x_training[i], y_training[i], c=colors[i], s=size, linewidths=0)
plt.show()
plt.close()

# Initialisation variables
network = Network()

# initial network results
x, y = mgrid[-output_weight:output_weight:nbins * 1j, -output_weight:output_weight:nbins * 1j]
z = []
for xi, yi in zip(x.flatten(), y.flatten()):
    output = network.feedforward([xi, yi])
    z.append(output)
z = array(z)
plt.pcolormesh(x, y, z.reshape(x.shape), shading='auto')
plt.show()
plt.close()

# Training
epochs = 100001
eta = 0.3
mini_batch_size = 100
network.SGD(training_data)
plt.close()

# initial network results
x, y = mgrid[-output_weight:output_weight:nbins * 1j, -output_weight:output_weight:nbins * 1j]
z = []
for xi, yi in zip(x.flatten(), y.flatten()):
    output = network.feedforward([xi, yi])
    z.append(output)
z = array(z)
plt.pcolormesh(x, y, z.reshape(x.shape), shading='auto')
plt.show()
plt.close()
