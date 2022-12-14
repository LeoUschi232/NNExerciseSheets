from matplotlib import pyplot as plt
from network import Network
from numpy import sin, cos, pi, mgrid, sqrt, vstack, array

# Define instance variables
debug = 1
training_size = 500
training_range = range(training_size)
loops = 10
test_size = 1600
test_range = range(test_size)
xy_limit = 2
size = 50
plot_training = 0
output_weight = 1

# Initialisation variables
network = Network([2, 8, 8, 1])

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
training_data = [[array(x_training[k], y_training[k]), expected_output[k]] for k in training_range]

# Testing data
l = int(sqrt(test_size))
x_test = [2 * xy_limit * ((k / test_size) - 0.5) for k in test_range]
y_test = [2 * xy_limit * ((k % l) - l / 2) / l for k in test_range]
test_data = [[x_test[k], y_test[k]] for k in test_range]


# Color function returns the color in dependence of the supposed quadrant
# for positive quadrants (1 = top-right, 3 = bottom-left) -> blue
# for negative quadrants (2 = top-left, 4 = bottom-right) -> yellow
def color_function(xy_sign: float):
    return 'red' if xy_sign < 0.0 else 'blue'


# initial network results
nbins = 1000
xi, yi = mgrid[-1:1:nbins * 1j, -1:1:nbins * 1j]
zi = network.feedforward(vstack([xi.flatten(), yi.flatten()]))
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
plt.show()
plt.close()

# Training
network.SGD(training_data=training_data,
            epochs=1000,
            mini_batch_size=10,
            eta=0.01)
plt.close()

# after training network results
xi, yi = mgrid[-1:1:nbins * 1j, -1:1:nbins * 1j]
zi = network.feedforward(vstack([xi.flatten(), yi.flatten()]))
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
plt.show()
plt.close()
