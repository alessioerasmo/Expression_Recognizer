from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tensorflow import keras

class generator:

    def __init__(self):

        self.operators_path = "Learning\Datasets\operators\\test"
        self.files = os.listdir(self.operators_path)
        for i in range(len(self.files)):
            if not self.files[i].endswith(".jpg"):
                self.files.remove(self.files[i])
        (x_train, y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()


    def pick_random_operator(self):
        index = random.randrange(0,len(self.files))
        img_path = self.operators_path + "\\" + self.files[index]
        labels = open(self.operators_path + "\labels_test.txt").read().split(",")
        if labels[int(self.files[index][:-4]) - 1] == "x":
            return [np.array(Image.open(img_path).convert("L")),"*" ]
        else:    
            return [np.array(Image.open(img_path).convert("L")),labels[int(self.files[index][:-4]) - 1] ]

    def pick_random_number(self):
        index = random.randrange(0, len(self.x_test))
        return [self.x_test[index], self.y_test[index]]

    def pick_random_expression(self, list):
        images = []
        values = ""
        for i in range(len(list)):
            for j in range(list[i]):
                num = self.pick_random_number()
                images.append(np.rot90(np.rot90(np.rot90(num[0]))))
                values += str(num[1])
            
            if (i+1 < len(list)):
                op = self.pick_random_operator()
                images.append(np.rot90(np.rot90(np.rot90(op[0]))))
                values += str(op[1])
        
        images = np.rot90(np.array(images).reshape(-1, 28))
        
        return (images, values)



gen = generator()


save_path = "Test\exports\\"

images_to_create = 50

labels = ""
for i in range(images_to_create):
    image_path = save_path + str(i) + ".jpg"

    
    array_length = random.randrange(2,4)
    expr_array = []
    
    for i in range(array_length):
        expr_array.append(random.randrange(1, 3))
    
    #expr_array = [2, 2]

    expression = gen.pick_random_expression(expr_array)
    Image.fromarray(expression[0]).save(image_path)
    labels = labels +  expression[1] + ","

file = open(save_path + "labels.txt", "w") 
file.write(labels[:-1])