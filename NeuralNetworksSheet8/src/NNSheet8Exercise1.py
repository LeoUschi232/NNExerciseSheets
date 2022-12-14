import mnist_loader
import network
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=2, suppress=True)
# This will load the MNIST data in the form expected by network.py, set up a
# network with 784 (28 pixels by 28 pixels) input neurons, 30 neurons in the hidden
# layer and 10 neurons in the output layer.

# This network has 30*784+10*30=23820 weights
# and 30+10=40 biases

# The last line trains the network on mini batches of size 10 in 10 epochs with a
# training parameter of 3.0. (As discussed in the lecture, Nielsen’s test_data should
# actually be called validation_data.) Play with these parameters and observe the
# performance of the network.
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
net.SGD(training_data=training_data,
        epochs=100,
        mini_batch_size=10,
        eta=3.0,
        test_data=test_data)

# You can extract an individual data sample, look at it and check the answer of the
# network with

xtest, ytest = test_data[0]
plt.imshow(xtest.reshape(28, 28), cmap='binary')
plt.show()
net.feedforward(xtest)

# If you create your own 28x28 greyscale image of a digit with Gimp and save it as
# raw data (the file size should be 784 bytes), you can ask your network to recognize
# the digit with (the /255 normalises the data to [0, 1] and the 1- makes white/empty
# pixels give an input value of 0 to the input neurons)
failed = "failed"
succeeded = "succeeded"
print(f"\nNow testing custom images:")
for i in range(10):
    my_digit = np.fromfile(f'../resources/MyDigit{i}.raw', dtype=np.ubyte)
    test = 1 - my_digit / 255.0
    output = net.feedforward(np.reshape(test, (784, 1)))
    print(f"\n{output.reshape(10)}")
    max_index = 0
    max_output = -1.0
    for index in range(len(output)):
        if output[index] > max_output:
            max_output = output[index]
            max_index = index
    print(f"Image {i} is the digit {max_index}")
    print(f"The assertion if image {i} {succeeded if i == max_index else failed}")
