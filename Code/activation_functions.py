import math
import numpy as np

def activation_logistic(input):
    return (1 + (math.e ** -input)) ** -1

'''Version of sigmoid in lab 1
def activation_testSigmoid(input):
    return 1 / (1 + np.exp(-input))
'''
def activation_relu(input):
    return np.maximum(0,input)

def activation_tanh(input):
    return math.tanh(input)

def activation_leaky_relu(input):
    return np.maximum(0.03*input,input)

def activation_elu(input):
    if input > 0:
        return input
    else:
        return (np.exp(input)-1)