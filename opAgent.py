from agents import *
from tensorflow import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class opAgent(Agent):

    def __init__(self,env=None):
        # loading the exported model
        self.model = keras.models.load_model('Learning/Model_Exports/operators.keras', safe_mode=False)
        # the environment is an array of images, each converted into a numpy array of shape (28, 28)
        if not env is None:
            self.set_env(env)

    def go(self):

        predictions = self.model.predict(self.env)

        signs = []
        values = np.zeros(len(predictions), dtype='int32') 
        for i in range(len(predictions)):
            max_prob = 0
            for j in range(4):
                if (predictions[i][j] >= max_prob):
                    max_prob = predictions[i][j]
                    values[i] = np.int32(j)
            if values[i] == 0:
                signs.append('+')
            elif values[i] == 1:
                signs.append('-')
            elif values[i] == 2:
                signs.append('*')
            elif values[i] == 3:
                signs.append('/')

        # outs an array with predictions as chars '+', '-', 'x', '/'
        return np.array(signs) 
    
    
    def set_env(self, new_env):
        self.env=np.array(new_env).reshape(-1, 28, 28, 1) 
        
    

# testing agent and plotting results 
def test(testing_examples):
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
    x_test = []
    for i in range(len(y_test)):
        img = np.array(Image.open("Learning\Datasets\operators/test/" + str(i+1) +".jpg").convert('L'))
        x_test.append(img)
    x_test = np.array(x_test)
    assert len(x_test) == len(y_test)

    import random
    random_int = random.randint(0, len(x_test)-testing_examples)
    examples_images = x_test[random_int:random_int+testing_examples]
    opagent = opAgent(examples_images)
    predictions = opagent.go()

    import matplotlib.pyplot as plt
    for i in range(len(examples_images)):
        plt.imshow(examples_images[i])
        plt.title("Value predicted: " + str(predictions[i]) )
        plt.show()

if __name__ == "__main__":
    test(15) #do testing with 5 random images