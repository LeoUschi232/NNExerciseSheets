import scipy.io as sio
from network import Network

# Lecture code
data = sio.loadmat('../resources/documents/training_and_validation-orig.mat')["affNISTdata"]
data = data[0, 0]
images = data[2].swapaxes(0, 1)
digitsvect = data[4].swapaxes(0, 1)
digitsint = data[5][0]
IMG_SIZE = len(images[0])
training_data = []
for i in range(50000):
    training_data.append(((images[i] / 255).reshape(-1, 1), digitsvect[i].reshape(-1, 1)))
test_data = []
for i in range(50000, 60000):
    test_data.append(((images[i] / 255).reshape(-1, 1), digitsint[i]))

# instance variables
net = Network([784, 12, 12, 12, 10])
epochs = 150
mini_batch_size = 10
eta = 0.01

# training the network
net.SGD(training_data=training_data,
        epochs=150,
        mini_batch_size=10,
        eta=0.01,
        test_data=test_data)
