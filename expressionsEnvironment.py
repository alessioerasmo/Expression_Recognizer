from agents import *
import numpy as np

class expressionsEnvironment(Environment):
    
    def __init__(self, image=None):
        if not image is None:
            self.initial_percepts(image)

    def initial_percepts(self, image):
        image = np.rot90(np.rot90(np.rot90(np.array(image))))
        self.images = image.reshape(-1, 28, 28)
        for i in range(len(self.images)):
            self.images[i] = np.rot90(self.images[i])

    def not_finished(self):
        return len(self.images) > 0

    def pick_next(self):

        if len(self.images) == 0:
            return None
        
        elif len(self.images)  == 1:
            image = self.images[0]
            self.images = []

        elif len(self.images)>1:
            image = self.images[0]
            self.images = self.images[1:]
         
        return image
    

def test():
    from PIL import Image
    import matplotlib.pyplot as plt

    image = np.array(Image.open("Test\\3.jpg").convert('L'))

    plt.imshow(image)
    plt.show()

    exp_env = expressionsEnvironment(image)

    while exp_env.not_finished():
        plt.imshow(exp_env.pick_next())
        plt.show()


if __name__ == "__main__":
    test()
