from agents import *
from tensorflow import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class classificationAgent(Agent):
    def __init__(self,env):
        # loading the exported model
        self.model = keras.models.load_model('Learning\Model_Exports\operators_numbers_classificator.h5')
        # the environment is an array of images, each converted into a numpy array of shape (28, 28)
        self.env=np.array(env).reshape(-1, 28, 28, 1) 

    def go(self):
        predictions = self.model.predict(self.env)
        labels = []
        for i in range(len(predictions)):
            if predictions[i] <= 0.5:
                labels.append("numero")
            else:
                labels.append("operazione")
        return np.array(labels)


# testing agent and plotting results 
def test(testing_examples):
    import random
    
    labels_file = open("Learning\Datasets\operators/test/labels_test.txt")
    labels = np.array(labels_file.read().split(","))   
    op_y_test = np.ones(len(labels))
    op_x_test = []
    for i in range(len(op_y_test)):
        img = np.array(Image.open("Learning\Datasets\operators/test/" + str(i+1) +".jpg").convert('L'))
        op_x_test.append(img)
    op_x_test = np.array(op_x_test)
    assert len(op_x_test) == len(op_y_test)
    
    cont_op_examples = int((testing_examples-(testing_examples%2))/2)
    cont_num_examples = testing_examples - cont_op_examples
    random_int = random.randint(0, len(op_x_test)-cont_op_examples)
    example_op = op_x_test[random_int:random_int+cont_op_examples]
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    random_int = random.randint(0, len(x_test)-cont_num_examples)
    example_num = x_test[random_int:random_int+cont_num_examples]

    examples_img = []
    for i in range(len(example_num)):
        examples_img.append(example_num[i])
    for i in range(len(example_op)):
        examples_img.append(example_op[i])

    examples_img = np.array(examples_img) 
    classificationagent = classificationAgent(examples_img)
    predictions = classificationagent.go()

    for i in range(len(examples_img)):
        plt.imshow(examples_img[i])
        plt.title("Value predicted: " + str(predictions[i]) )
        plt.show()


if __name__ == "__main__":
    test(20) #do testing with 20 random images 