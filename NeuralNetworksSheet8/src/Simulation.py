from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from numpy import sin, cos, pi, mgrid, array

print(f"Simulation of advanced neural network started started")

# Creating a machine learning model to simulate the neural network that will learn my prompt
input_shape = (2,)
print(f"Simulation of advanced neural network at checkpoint 1")
network = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation="sigmoid"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="sigmoid"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
    layers.Dense(1, activation="softmax")
])
print(f"Simulation of advanced neural network at checkpoint 2")
network.summary()

print(f"Simulation of advanced neural network at checkpoint 3")

# Define training variables
training_size = 1000
training_range = range(training_size)
loops = 10
xy_limit = 2

# Training data loops
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
training_input = [(x_training[k], y_training[k]) for k in training_range]
expected_output = [xy_limit if k < (training_size / xy_limit) else -xy_limit for k in training_range]
training_data = [[training_input[k], expected_output[k]] for k in training_range]

print(f"Simulation of advanced neural network at checkpoint 4")


# Color function to ascertain the point's value
def color_function(xy_sign: float):
    return 'red' if xy_sign < 0.0 else 'blue'


# Plotting the training function
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

print(f"Simulation of advanced neural network at checkpoint 5")

# Initialisation variables
nbins = 100

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

# Training the machine learning network
network.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
network.fit(training_input, expected_output,
            batch_size=50,
            epochs=1001,
            validation_split=0.1)

print(f"Simulation of advanced neural network at checkpoint 6")

# after training network results
z = []
for xi, yi in zip(x.flatten(), y.flatten()):
    output = network.output(xi, yi)
    z.append(output)
z = array(z)
plt.pcolormesh(x, y, z.reshape(x.shape), shading='auto')
plt.show()
plt.close()

print(f"Simulation of advanced neural network finished")
