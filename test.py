from expressionsAgent import expressionAgent
from expressionsEnvironment import expressionsEnvironment
import numpy as np
from PIL import Image


labels = open("Test\exports\labels.txt").read().split(",")

errors = 0

exp_env = expressionsEnvironment()
expr_ag = expressionAgent()

for i in range(len(labels)):
    img = np.array(Image.open("Test\exports\\" + str(i) + ".jpg").convert('L'))

    exp_env.initial_percepts(img)
    expr_ag.set_env(exp_env)
    
    pred = expr_ag.go(20)

    if pred!=labels[i]:
        errors += 1

print(errors," errors on ",len(labels), " total examples")

"""
import matplotlib.pyplot as plt
plt.imshow(img)

plt.title(pred)
plt.show()
"""