from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
model2 = keras.Sequential(
    [layers.Input(shape=784),
     layers.Dense(30, activation="sigmoid"),
     layers.Dense(10, activation="sigmoid"),
     ])
model2.summary()
model2.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
model2.fit(x_train.reshape(-1, 784), y_train, batch_size=10, epochs=10,
           validation_split=0.1)
