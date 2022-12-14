from numpy import exp, random, dot, sin, zeros, array, warnings, VisibleDeprecationWarning, isnan, ones
from sys import exit

warnings.filterwarnings('ignore', category=VisibleDeprecationWarning)


class Network(object):
    def __init__(self, sizes=None):
        if sizes is None:
            sizes = [4, 12, 12, 12, 1]
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [random.randn(y) for y in sizes[1:]]
        self.weights = [random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.b = array([array([random.uniform(-1.0, 1.0)
                               for i in range(sizes[l])])
                        for l in range(len(sizes))][1:])
        self.w = array([array([array([random.uniform(-1.0, 1.0)
                                      for j in range(sizes[l - 1])])
                               for i in range(sizes[l])])
                        for l in range(len(sizes))][1:])

    def feedforward(self, throughput):
        for b, w in zip(self.biases, self.weights):
            throughput = sigmoid(dot(w, throughput) + b)
        return throughput

    def output(self, x, y):
        throughput = array([x, y, sin(x), sin(y)])
        return self.feedforward(throughput)

    def cost(self, training_data):
        cost = 0.0
        for input, expected_output in training_data:
            x, y = input
            summand = expected_output - self.output(x, y)
            cost += (summand * summand)
        return cost[0]  # noqa

    def cost_derivative(self, training_set):
        input, expected_output = training_set
        x, y = input
        return self.output(x, y) - expected_output

    def backprop(self, training_set):
        nabla_b = [zeros(b.shape) for b in self.biases]
        nabla_w = [zeros(w.shape) for w in self.weights]
        input, expected_output = training_set
        x, y = input
        throughput = array([x, y, sin(x), sin(y)])
        throughputs = [throughput]
        zl = []
        for b, w in zip(self.biases, self.weights):
            z = dot(w, throughput) + b
            zl.append(z)
            throughput = sigmoid(z)
            throughputs.append(throughput)
        error = self.cost_derivative(training_set) * sigmoid_derivative(zl[-1])
        nabla_b[-1] = error
        nabla_w[-1] = error.reshape(-1, 1) @ throughputs[-2].reshape(1, -1)

        for l in range(2, self.num_layers):
            z = zl[-l]
            sg = sigmoid_derivative(z)
            w = self.weights[-l + 1].transpose()
            w_error = (w @ error)
            error = w_error * sg
            nabla_b[-l] = error
            nabla_w[-l] = error.reshape(-1, 1) @ throughputs[-l - 1].reshape(1, -1)
        return nabla_b, nabla_w

    def gradient_descent(self, training_data, epochs=10001, eta=0.3, epoch_output=10):
        n = len(training_data)
        eta_b = array([array([eta for i in range(len(self.biases[l]))])
                       for l in range(len(self.biases))])
        eta_w = array([array([array([eta for j in range(len(self.weights[l][i]))])
                              for i in range(len(self.weights[l]))])
                       for l in range(len(self.weights))])

        previous_cost = self.cost(training_data)
        print(f"\nCost before training: {previous_cost}\n")
        cost_counter = 0

        for epoch in range(epochs):
            random.shuffle(training_data)
            for training_set in training_data:
                nabla_b, nabla_w = self.backprop(training_set)
                self.biases -= eta_b * nabla_b
                self.weights -= eta_w * nabla_w

            if epoch % epoch_output == 0:
                previous_cost, cost_counter, should_break = \
                    self.epoch_variables(epoch, training_data, eta, previous_cost, cost_counter)

    def epoch_variables(self, epoch, training_data, eta, previous_cost, cost_counter):
        should_break = False
        current_cost = self.cost(training_data)
        print(f"Epoch: {epoch} -> Cost after training: {current_cost}")
        if current_cost < 4.0:
            should_break = True
        if abs(previous_cost - current_cost) < 0.01:
            cost_counter += 1
            if cost_counter > 10:
                self.jump_out_of_hole(training_data, eta, current_cost)
                cost_counter = 0
        else:
            cost_counter = 0
        previous_cost = current_cost
        return previous_cost, cost_counter, should_break

    def jump_out_of_hole(self, training_data, eta, current_cost):
        i = 1
        while abs(current_cost - self.cost(training_data)) < 0.01:
            big_eta = (10.0 ** i) * eta
            big_eta_b = array([array([big_eta for i in range(len(self.biases[l]))])
                               for l in range(len(self.biases))])
            big_eta_w = array([array([array([big_eta for j in range(len(self.weights[l][i]))])
                                      for i in range(len(self.weights[l]))])
                               for l in range(len(self.weights))])

            for training_set in training_data:
                nabla_b, nabla_w = self.backprop(training_set)
                self.biases -= big_eta_b * nabla_b
                self.weights -= big_eta_w * nabla_w
            i += 1


# Miscellaneous functions
def sigmoid(z):
    expz1 = exp(z)
    if isnan(expz1).any():
        expz1 = ones(shape=z.shape)
    return 1.0 / (1.0 + expz1)


def sigmoid_derivative(z):
    expz1 = exp(z)
    if isnan(expz1).any():
        expz1 = ones(shape=z.shape)
    expz2 = 1 / (1 + expz1)
    return - expz1 * expz2 * expz2


def random_sign():
    return (-1) ** random.randint(0, 1)


def printf(var):
    print(f"Variable:\n{var}\n")
    exit(0)
