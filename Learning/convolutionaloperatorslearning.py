import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



# reading y_train
labels_file = open("Learning/Datasets/operators/labels.txt")
labels = np.array(labels_file.read().split(","))

y_train=[]
for i in range(len(labels)):
    if (labels[i] == '+'):
        y_train.append(0)
    if (labels[i] == '-'):
        y_train.append(1)
    if (labels[i] == 'x'):
        y_train.append(2)
    if (labels[i] == '/'):
        y_train.append(3)
y_train = np.array(y_train)

# reading x_train
x_train = []
for i in range(len(y_train)):
    img = np.array(Image.open("Learning/Datasets/operators/" + str(i+1) +".jpg").convert('L'))
    x_train.append(img)


x_train = np.array(x_train)
assert len(x_train)== len(y_train)

print("\n\ntraining input sizes : \n\nx_train shape: ", x_train.shape, "\ny_train shape: ", y_train.shape)


# reading y_test
labels_file = open("Learning/Datasets/operators/test/labels_test.txt")
labels = np.array(labels_file.read().split(","))

y_test=[]
for i in range(len(labels)):
    if (labels[i] == '+'):
        y_test.append(0)
    if (labels[i] == '-'):
        y_test.append(1)
    if (labels[i] == 'x'):
        y_test.append(2)
    if (labels[i] == '/'):
        y_test.append(3)
y_test = np.array(y_test)

# reading x_test
x_test = []
for i in range(len(y_test)):
    img = np.array(Image.open("Learning/Datasets/operators/test/" + str(i+1) +".jpg").convert('L'))
    x_test.append(img)

x_test = np.array(x_test)
assert len(x_test) == len(y_test)

print("\n\ntesting input sizes : \n\nx_test shape: ", x_test.shape, "\ny_test shape: ", y_test.shape)


# Feed-Forward neural network model generation

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda

from keras import layers
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Lambda(lambda x : x/255.0))
model.add(layers.Flatten())
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dense(4))
model.add(Activation('softmax')) # sum of all output features equals to 1, this means that the predictor outputs the density of probability of each number

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) 

model.summary()

# learning
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

from plot_history import *
plot_history(history)

# testing model 
loss, accuracy = model.evaluate(x_test,y_test)
print("\n\nTesting results:\n\nloss: ", loss, "\naccuracy", accuracy,"\n\n")

# export model
# model.save('Learning/Model_Exports/operators.keras')
