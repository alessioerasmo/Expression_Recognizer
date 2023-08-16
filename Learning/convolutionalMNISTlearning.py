import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda

import matplotlib.pyplot as plt
import random as rdm
import numpy as np
import matplotlib.image as mpimg


# load data
tf.keras.datasets.mnist.load_data(path="mnist.npz")

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

from keras import layers
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(Lambda(lambda x : x/255.0))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.add(Activation('softmax')) # sum of all output features equals to 1, this means that the predictor outputs the density of probability of each number

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) 

model.summary()

# learning
model.fit(x_train, y_train, epochs=40)

# testing model
loss, accuracy = model.evaluate(x_test,y_test)
print("\n\nTesting results:\n\nloss: ", loss, "\naccuracy", accuracy,"\n\n")


# export model
model.save('Learning/Model_Exports/MNIST.keras')
