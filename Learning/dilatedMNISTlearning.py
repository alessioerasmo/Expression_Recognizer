import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda

import matplotlib.pyplot as plt
import random as rdm
import numpy as np
import matplotlib.image as mpimg





tf.keras.datasets.mnist.load_data(path="mnist.npz")

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# we can apply dilation to help the model to recognize thicker numbers, this led the model to improve his accuracy on test data 
from skimage import morphology as mrph
from skimage.morphology import square

dilation_structural_element = np.array([[0,1,0],
                                        [1,1,1],
                                        [0,1,0]])

# new training data
x_train_thick = []
for i in range(len(x_train)):
    x_train_thick.append(mrph.dilation(x_train[i], dilation_structural_element))

x_train_thick = np.array(x_train_thick)

new_x_train = np.array( [x_train, x_train_thick]).reshape(120000, 28, 28)
new_y_train = np.array([y_train, y_train]).reshape(120000)

"""

# showing some imgs for example
from random import Random

for i in range(10):
    index = int(Random().random()*60000)
    img1 = np.array([new_x_train[index], new_x_train[index+60000]]).reshape(56,28)
    plt.imshow(img1)
    plt.show()

"""

# Feed-Forward neural network model generation

model = Sequential()
model.add(Flatten(input_shape=(28,28,)))
model.add(Lambda(lambda x : x/255.0))
model.add(Dense(1000))
model.add(Activation('relu')) 
model.add(Dense(100))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax')) # sum of all output features equals to 1, this means that the predictor outputs the density of probability of each number

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) 

model.summary()

# learning
model.fit(new_x_train, new_y_train, epochs=50)

# testing model
loss, accuracy = model.evaluate(x_test,y_test)
print("\n\nTesting results:\n\nloss: ", loss, "\naccuracy", accuracy,"\n\n")

# export model
# model.save('Learning/Model_Exports/MNIST.keras')

"""
LAST EXPORTED MODEL TEST RESULTS:

*.h5
    loss:  0.054929010570049286
    accuracy 0.9815000295639038

*.keras
    loss:  0.05537978559732437
    accuracy 0.9810000061988831

"""