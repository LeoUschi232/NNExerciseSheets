from network import Network
from numpy import mgrid, vstack, array, sign
from matplotlib import pyplot as plt
import random

# instance variables
net = Network([2, 4, 1])
nbins = 100
training_size = 2000
training_range = range(training_size)

# initial network results
xi, yi = mgrid[-1:1:nbins * 1j, -1:1:nbins * 1j]
zi = net.feedforward(vstack([xi.flatten(), yi.flatten()]))
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
plt.show()
plt.close()

# training the network
training_data = []
for i in training_range:
    x = random.uniform(-1.01, 1.01)
    y = random.uniform(-1.01, 1.01)
    training_data.append((array([[x], [y]]), array([[1 / 2 + sign(x * y) / 2]])))
net.SGD(training_data=training_data,
        epochs=10001,
        mini_batch_size=20,
        eta=0.3)

# after training network results
xi, yi = mgrid[-1:1:nbins * 1j, -1:1:nbins * 1j]
zi = net.feedforward(vstack([xi.flatten(), yi.flatten()]))
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
plt.show()
plt.close()
