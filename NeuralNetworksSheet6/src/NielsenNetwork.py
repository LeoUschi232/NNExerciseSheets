"""
network.py
~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

# some changes for python3 by SR (incl. list() in mnist_loader)

# Libraries
# Standard library
import random
# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

class Network(object):

    def __init__(self, sizes=None):
        if sizes is None:
            sizes = [4, 12, 12, 12, 1]
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, throughput):
        throughput = [[throughput[0]], [throughput[1]], [np.sin(throughput[0])], [np.sin(throughput[1])]]
        for b, w in zip(self.biases, self.weights):
            throughput = sigmoid(np.dot(w, throughput) + b)
        return throughput

    def output(self, input):
        return self.feedforward(input)[0][0]

    def cost(self, training_data):
        cost = 0.0
        for input, expected_output in training_data:
            x, y = input
            summand = expected_output - self.output((x, y))
            cost += (summand * summand)
        return cost

    def SGD(self, training_data,
            epochs=10001,
            mini_batch_size=0,
            eta=0.02,
            epoch_output=10):
        if mini_batch_size <= 0:
            mini_batch_size = len(training_data)
        n = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if epoch % epoch_output == 0:
                current_cost = self.cost(training_data)
                print(f"Epoch: {epoch} -> Cost after training: {current_cost}")

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for input, expected_output in mini_batch:
            x, y = input
            throughput = np.array([[x], [y], [np.sin(x)], [np.sin(y)]])
            delta_nabla_b, delta_nabla_w = self.backprop(throughput, expected_output)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, throughput, expected_output):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = throughput
        activations = [activation]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = cost_derivative(activations[-1], expected_output) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


# Miscellaneous functions
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def cost_derivative(output_activations, y):
    return output_activations - y


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
