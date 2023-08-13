# classi progetto   
from expressionsAgent import *                       
from expressionsEnvironment import *            

from PIL import Image


labels = open("Test/exports/labels.txt").read().split(",")

errors = 0

exp_env = expressionsEnvironment()
expr_ag = expressionAgent()

for i in range(len(labels)):
    img = Image.open("Test/exports/" + str(i) + ".jpg")

    exp_env.initial_percepts(img)
    expr_ag.set_env(exp_env)
    
    pred = expr_ag.go(20)

    if pred!=labels[i]:
        errors += 1


print("\n",errors," errors on ",len(labels), " total examples\n",((len(labels)-errors)/len(labels))*100,"% of accuracy")