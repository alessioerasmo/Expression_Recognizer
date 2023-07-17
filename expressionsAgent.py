from agents import *
import numpy as np


from expressionsEnvironment import expressionsEnvironment
from numAgent import numAgent
from opAgent import opAgent 
from classificationAgent import classificationAgent


class expressionAgent(Agent):
    def __init__(self,env=None):
        # the environment is an object of type "expressionsEnvironment"
        self.opagent = opAgent()
        self.numAgent = numAgent()
        self.classifier = classificationAgent()
        if not env is None:
            self.set_env(env)

    def go(self,n=10, evaluate=False):
        """acts for n time steps"""

        expression = ""

        if self.env is None:
            return expression
        
        self.number_expected = True    
        for i in range(n):
            if self.env.not_finished():
                image = self.env.pick_next()
                if self.number_expected:
                    expression = expression + str(self.predict_number(image))
                else:
                    expression = expression + str(self.guess(image))
            else:
                if (evaluate):
                    return eval(expression)
                else: 
                    return expression
        
        if (evaluate):
            return eval(expression)
        else: 
            return expression
        
        
    def set_env(self, env):
        self.env = env

    def predict_number(self, image):
        self.number_expected = False
        self.numAgent.set_env(image)
        return self.numAgent.go()[0]

    def predict_operation(self, image):
        self.number_expected = True
        self.opagent.set_env(image)
        return self.opagent.go()[0]
    
    def guess(self, image):

        self.classifier.set_env(image)
        image_type = self.classifier.go()
        if (image_type == 'operazione'):
            value = self.predict_operation(image)
        else:
            value = self.predict_number(image)
        return value
                

def test():

    from PIL import Image
    import matplotlib.pyplot as plt

    # img = np.array(Image.open("Test\\10.jpg").convert('L'))
    img = np.array(Image.open("Test/exports/5.jpg").convert('L'))
    exp_env = expressionsEnvironment(img)
    agent = expressionAgent(exp_env)
    result = agent.go()

    plt.imshow(img)
    plt.title("expression found: " + str(result) + " = " + str(eval(result)))
    plt.show()


if __name__ == "__main__":
    test()
