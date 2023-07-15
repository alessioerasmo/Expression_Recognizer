from agents import *
from tensorflow import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class numAgent(Agent):
    def __init__(self,env=None):
        # loading the exported model
        self.model = keras.models.load_model('Learning\Model_Exports\MNIST.keras')
        # the environment is an array of images, each converted into a numpy array of shape (28, 28)
        if not env is None:
            self.set_env(env)

    def go(self):
        predictions = self.model.predict(self.env)
        values = np.zeros(len(predictions), dtype='int32') 
        for i in range(len(predictions)):
            max_prob = 0
            for j in range(10):
                if (predictions[i][j] >= max_prob):
                    max_prob = predictions[i][j]
                    values[i] = np.int32(j)
        # outs an array with predictions as integers 
        return values 
    
    
    def set_env(self, new_env):
        self.env=np.array(new_env).reshape(-1, 28, 28, 1) 
        
    

# testing agent and plotting results 
def test(testing_examples):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    import random
    random_int = random.randint(0, len(x_test)-testing_examples)
    examples_images = x_test[random_int:random_int+testing_examples]
    numagent = numAgent(examples_images)
    predictions = numagent.go()

    import matplotlib.pyplot as plt
    for i in range(len(examples_images)):
        plt.imshow(examples_images[i])
        plt.title("Value predicted: " + str(predictions[i]) )
        plt.show()


if __name__ == "__main__":
    test(5) #do testing with 5 random images