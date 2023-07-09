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

op_y_train = np.ones(labels.shape, dtype='int32')


op_x_train = []
for i in range(len(op_y_train)):
    img = np.array(Image.open("Learning\Datasets\operators/" + str(i+1) +".jpg").convert('L'))
    op_x_train.append(img)

op_x_train = np.array(op_x_train)

# learning_numbers

tf.keras.datasets.mnist.load_data(path="mnist.npz")

(num_x_train, num_y_train), (num_x_test, num_y_test) = keras.datasets.mnist.load_data()
assert num_x_train.shape == (60000, 28, 28)
assert num_x_test.shape == (10000, 28, 28)
assert num_y_train.shape == (60000,)
assert num_y_test.shape == (10000,)

numbers = len(op_x_train)*2
num_x_train = num_x_train[:numbers]

num_y_train = np.zeros(len(num_x_train), dtype='int32')

print(op_x_train.shape)
print(op_y_train.shape)
print(num_x_train.shape)
print(num_y_train.shape)

x_train = [op_x_train,num_x_train]
x_train = np.array(x_train)

#x_train = np.array([ op_x_train, num_x_train ])
#y_train = np.array([ op_y_train, num_y_train ]).reshape(-1)

#print(x_train.shape)
#print(y_train.shape)
