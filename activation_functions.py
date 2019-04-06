import numpy as np

class activation_functions(object):

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def relu(x):
        return np.max(x,0)


class activation_function_derivations(object):

    @staticmethod
    def dsigmoid(x):
        return x * (1 - x)

    @staticmethod
    def dtanh(x):
        return 1 - x**2

    @staticmethod
    def drelu(x):
        if (x < 0):
            return 0
        else:
            return 1
