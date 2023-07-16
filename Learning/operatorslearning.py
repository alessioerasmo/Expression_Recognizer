import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



# reading y_train
labels_file = open("Learning\Datasets\operators/labels.txt")
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
    img = np.array(Image.open("Learning\Datasets\operators/" + str(i+1) +".jpg").convert('L'))
    x_train.append(img)


x_train = np.array(x_train)
assert len(x_train)== len(y_train)

print("\n\ntraining input sizes : \n\nx_train shape: ", x_train.shape, "\ny_train shape: ", y_train.shape)


# reading y_test
labels_file = open("Learning\Datasets\operators/test/labels_test.txt")
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
    img = np.array(Image.open("Learning\Datasets\operators/test/" + str(i+1) +".jpg").convert('L'))
    x_test.append(img)

x_test = np.array(x_test)
assert len(x_test) == len(y_test)

print("\n\ntesting input sizes : \n\nx_test shape: ", x_test.shape, "\ny_test shape: ", y_test.shape)


# Feed-Forward neural network model generation

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda

model = Sequential()
model.add(Flatten(input_shape=(28,28,)))
model.add(Lambda(lambda x : x/255.0))
model.add(Dense(200))
model.add(Activation('sigmoid'))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dense(4))
model.add(Activation('softmax')) # sum of all output features equals to 1, this means that the predictor outputs the density of probability of each number

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) 

model.summary()

# learning
model.fit(x_train, y_train, epochs=500)

# testing model 
loss, accuracy = model.evaluate(x_test,y_test)
print("\n\nTesting results:\n\nloss: ", loss, "\naccuracy", accuracy,"\n\n")

# export model
# model.save('Learning/Model_Exports/operators.keras')

"""
LAST EXPORTED MODEL TEST RESULTS:

*.h5
    loss:  0.10491247475147247
    accuracy 0.9833333492279053

*.keras
    loss:  0.0963006317615509
    accuracy 0.9888888597488403

"""


"""
# troviamo le immagini sbagliate
def max_prob(arr):
    maxind = 0
    for i in range(len(arr)):
        if arr[i] >= arr[maxind]:
            maxind = i
    return maxind

predictions = model.predict(x_test)
prediction_values = []

for i in range(len(predictions)):
    prediction_values.append(max_prob(predictions[i]))

wrong_numbers = []

for i in range(len(y_test)):
    if prediction_values[i] != y_test[i]:
        plt.imshow(x_test[i])
        plt.text(0,0, "Predicted "+ str(prediction_values[i]) + " instead of " + str(y_test[i]), color="red", backgroundcolor="black" )
        plt.show()
        wrong_numbers.append(i+1)  

print(len(wrong_numbers)," wrong numbers:\n",wrong_numbers)

print(model.predict(np.array(x_test[9]).reshape(-1, 28, 28, 1)))

"""