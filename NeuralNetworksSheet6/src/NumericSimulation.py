from matplotlib import pyplot as plt
from NumericNetwork import Network
from numpy import sin, cos, pi, sqrt

# Define instance variables
training_size = 1000
training_range = range(training_size)
loops = 10
xy_limit = 2

# Initialisation variables
network = Network()

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
training_data = [[x_training[k], y_training[k], expected_output[k]] for k in training_range]

# Testing variables
test_size = 1600
test_range = range(test_size)

# Testing data
l = int(sqrt(test_size))
x_test = [xy_limit * xy_limit * ((k / test_size) - 0.5) for k in test_range]
y_test = [xy_limit * xy_limit * ((k % l) - l / xy_limit) / l for k in test_range]
test_data = [[x_test[k], y_test[k]] for k in test_range]


# Color function returns the color in dependence of the point's
def color_function(xy_sign: float):
    return 'red' if xy_sign < 0.0 else 'blue'


size = 50
colors = []
for i in training_range:
    colors.append(color_function(expected_output[i]))
for i in training_range:
    plt.scatter(x_training[i], y_training[i], c=colors[i], s=size, linewidths=0)
plt.show()
plt.close()

# Output before testing
colors = []
for i in test_range:
    colors.append(color_function(network.output([x_test[i], y_test[i]])))
for i in test_range:
    plt.scatter(x_test[i], y_test[i],
                c=colors[i],
                s=size,
                linewidths=0)
plt.show()

# Training
network.gradient_descent(training_data=training_data,
                         epochs=1001,
                         minibatch_size=20,
                         eta=0.02,
                         epoch_output=1)
plt.close()

# Testing
colors = []
for i in test_range:
    colors.append(color_function(network.output([x_test[i], y_test[i]])))
for i in test_range:
    plt.scatter(x_test[i], y_test[i],
                c=colors[i],
                s=size,
                linewidths=0)
plt.show()
