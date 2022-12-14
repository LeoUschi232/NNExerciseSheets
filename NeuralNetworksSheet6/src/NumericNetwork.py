from numpy import exp, random, dot, sin, zeros
from sys import exit


class Network(object):
    def __init__(self, sizes=None):
        if sizes is None:
            sizes = [4, 12, 12, 1]
        self.num_ls = len(sizes)
        self.sizes = sizes
        arrayed_biases = [random.randn(y, 1) for y in sizes[1:]]
        self.b = [[random_sign() * arrayed_biases[i1][i2][0] for i2 in range(len(arrayed_biases[i1]))] for i1 in
                  range(len(arrayed_biases))]
        arrayed_weights = [random.rand(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.w = [[[random_sign() * arrayed_weights[l][i][j]
                    for j in range(len(arrayed_weights[l][i]))]
                   for i in range(len(arrayed_weights[l]))]
                  for l in range(len(arrayed_weights))]
        self.w0 = self.w[-1][0]
        self.hidden_ls = range(len(arrayed_weights) - 1)

    def z(self, l: int, throughput: list):
        b = self.b
        w = self.w
        sum_l = zeros(len(b[l]))
        neurons = range(len(sum_l))
        for i in neurons:
            sum_l[i] = dot(w[l][i], throughput)
        zl = [sum_l[i] + b[l][i] for i in neurons]
        return zl

    def sigmoid(self, l: int, throughput: list):
        zl = self.z(l, throughput)
        sigmoid = 1 / (1 + exp(zl))
        return sigmoid

    def output(self, throughput: list):
        x, y = throughput
        throughput = [x, y, sin(x), sin(y)]
        for l in self.hidden_ls:
            throughput = self.sigmoid(l, throughput)
        output_sum = dot(self.w0, throughput)
        return output_sum

    def cost(self, training_data):
        cost = 0.0
        for xyz in training_data:
            x, y, expected_output = xyz
            summand = expected_output - self.output([x, y])
            cost += (summand * summand)
        return cost

    def dwlij_cost(self, training_data, l: int, i: int, j: int, h=0.002):
        self.w[l][i][j] += (h / 2)
        c_plus = self.cost(training_data)
        self.w[l][i][j] -= h
        c_minus = self.cost(training_data)
        self.w[l][i][j] += (h / 2)
        dwlij_cost = (c_plus - c_minus) / h
        return dwlij_cost

    def db_cost(self, training_data, l: int, i: int, h=0.002):
        self.b[l][i] += (h / 2)
        c_plus = self.cost(training_data)
        self.b[l][i] -= h
        c_minus = self.cost(training_data)
        self.b[l][i] += (h / 2)
        db_cost = (c_plus - c_minus) / h
        return db_cost

    def dw0_cost(self, training_data, j: int, h=0.002):
        self.w0[j] += (h / 2)
        c_plus = self.cost(training_data)
        self.w0[j] -= h
        c_minus = self.cost(training_data)
        self.w0[j] += (h / 2)
        dw0_cost = (c_plus - c_minus) / h
        return dw0_cost

    def gradient_descent(self, training_data, epochs=10001, minibatch_size=10, eta=0.02, epoch_output=1):
        w = self.w
        w0 = self.w0
        n = len(training_data)
        print(f"\nCost before training: {self.cost(training_data)}\n")
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + minibatch_size] for k in range(0, n, minibatch_size)]
            for mini_batch in mini_batches:
                for l in self.hidden_ls:
                    for i in range(len(w[l])):
                        for j in range(len(w[l][i])):
                            dwlij_cost = self.dwlij_cost(mini_batch, l, i, j)
                            self.w[l][i][j] -= eta * dwlij_cost
                        db_cost = self.db_cost(mini_batch, l, i)
                        self.b[l][i] -= eta * db_cost
                for j in range(len(w0)):
                    dw0_cost = self.dw0_cost(mini_batch, j)
                    self.w0[j] -= eta * dw0_cost
            cost_after_training = self.cost(training_data)
            if epoch % epoch_output == 0:
                print(f"Epoch: {epoch} -> Cost after training: {cost_after_training}")
            if cost_after_training < 4.0:
                break
        self.w[-1][0] = self.w0
        print(f"\nw:\n{self.w}")
        print(f"b:\n{self.b}")
        print(f"w0:\n{self.w0}")


def random_sign():
    return (-1) ** random.randint(0, 1)


def printf(var):
    print(f"Variable:\n{var}\n")
    exit(0)
