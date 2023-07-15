import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda

# learning_operations
labels_file = open("Learning\Datasets\operators/labels.txt")
labels = np.array(labels_file.read().split(","))

op_y_train = np.ones(labels.shape, dtype='int32')   # operator labeled as 1


op_x_train = []
for i in range(len(op_y_train)):
    img = np.array(Image.open("Learning\Datasets\operators/" + str(i+1) +".jpg").convert('L'))
    op_x_train.append(img)

op_x_train = np.array(op_x_train)

# learning_numbers
numbers_over_ops = 3    # relationship between numbers and operations in the dataset

tf.keras.datasets.mnist.load_data(path="mnist.npz")

(num_x_train, num_y_train), (num_x_test, num_y_test) = keras.datasets.mnist.load_data()
assert num_x_train.shape == (60000, 28, 28)
assert num_x_test.shape == (10000, 28, 28)
assert num_y_train.shape == (60000,)
assert num_y_test.shape == (10000,)

numbers = len(op_x_train)*numbers_over_ops

import random
index = random.randrange(0, len(num_x_train)-numbers)

num_x_train = num_x_train[index:index+numbers].reshape(-1, len(op_y_train), 28, 28)
num_y_train = np.zeros(len(op_y_train)*numbers_over_ops, dtype='int32').reshape(-1, len(op_y_train))    # number labeled as 0

x_train = [op_x_train]
y_train = [op_y_train]
for i in range(len(num_x_train)):
    x_train.append(num_x_train[i])
    y_train.append(num_y_train[i])

x_train = np.array(x_train).reshape(-1, 28, 28)
y_train = np.array(y_train).reshape(-1)

print("training with x_train of shape : ",x_train.shape,"\ntraining with y_test of shape : ",y_train.shape,"numbers over operators: ",numbers_over_ops)


# Feed-Forward neural network model generation

model = Sequential()
model.add(Flatten(input_shape=(28,28,)))
model.add(Lambda(lambda x : x/255.0))
model.add(Dense(200))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy']) 

model.summary()

model.fit(x_train, y_train, epochs=500)



# model test
labels_file = open("Learning\Datasets\operators/test/labels_test.txt")
labels = np.array(labels_file.read().split(","))

op_y_test = np.ones(labels.shape)

op_x_test = [] 
for i in range(len(op_y_test)):
    img = np.array(Image.open("Learning\Datasets\operators/test/" + str(i+1) +".jpg").convert('L'))
    op_x_test.append(img)

op_x_test = np.array(op_x_test)

x_test = np.array([op_x_test, num_x_test[1000:1000+len(op_x_test)]]).reshape(-1, 28, 28)
y_test = np.array([op_y_test, np.zeros(op_y_test.shape, dtype='int32')]).reshape(-1)

loss, accuracy = model.evaluate(x_test,y_test)
print("\n\nTesting results with test of 50% numbers and  50% operators of length:",len(x_test),"\n\nloss: ", loss, "\naccuracy", accuracy,"\n\n")

loss, accuracy = model.evaluate(num_x_test, np.zeros(num_y_test.shape, dtype='int32'))
print("\n\nTesting results with MNIST numbers set:\n\nloss: ", loss, "\naccuracy", accuracy,"\n\n")

loss, accuracy = model.evaluate(op_x_test, op_y_test)
print("\n\nTesting results with only operators set:\n\nloss: ", loss, "\naccuracy", accuracy,"\n\n")


# export model
model.save('Learning\Model_Exports/operators_numbers_classificator.keras')

"""
LAST EXPORTED MODEL TEST RESULTS:

*.h5

    Testing results with test of 50% numbers and  50% operators of length: 360

    loss:  0.03594835475087166
    accuracy 0.9611111283302307


    313/313 [==============================] - 0s 657us/step - loss: 0.0350 - accuracy: 0.9590


    Testing results with MNIST numbers set:

    loss:  0.03496096655726433
    accuracy 0.9589999914169312


    6/6 [==============================] - 0s 801us/step - loss: 0.0341 - accuracy: 0.9611


    Testing results with only operators set:

    loss:  0.03406630456447601
    accuracy 0.9611111283302307

*.keras

    Testing results with test of 50% numbers and  50% operators of length: 360

    loss:  0.03263424336910248
    accuracy 0.9555555582046509


    313/313 [==============================] - 0s 662us/step - loss: 0.0146 - accuracy: 0.9840


    Testing results with MNIST numbers set:

    loss:  0.014599429443478584
    accuracy 0.984000027179718


    6/6 [==============================] - 0s 801us/step - loss: 0.0538 - accuracy: 0.9222


    Testing results with only operators set:

    loss:  0.05380188673734665
    accuracy 0.9222221970558167
    
"""