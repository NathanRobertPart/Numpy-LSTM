import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.max(x,0)

def dsigmoid(x):
    return x * (1 - x)

def dtanh(x):
    return 1 - x**2

def drelu(x):
    if (x < 0):
        return 0
    else:
        return 1
