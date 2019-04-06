import numpy as np
from activation_functions import activation_functions as af

class neuron(object):

    def __init__(self,H_size,z_size):
        self.initialise_weights(H_size,z_size)

    def initialise_weights(self,H_size,z_size):
        self.forget_weight = np.random.randn(H_size,z_size)
        self.input_weight = np.random.randn(H_size,z_size)
        self.cell_weight = np.random.randn(H_size,z_size)
        self.output_weight = np.random.randn(H_size,z_size)

        self.forget_bias = np.random.randn(H_size,1)
        self.input_bias = np.random.randn(H_size, 1)
        self.cell_bias = np.random.randn(H_size, 1)
        self.output_bias = np.random.randn(H_size, 1)

    def forget_gate(self,z):
        return af.sigmoid(np.dot(self.forget_weight,z) + self.forget_bias)

    def input_gate(self,z):
        it = af.sigmoid(np.dot(self.input_weight,z) + self.input_bias)
        ct = af.tanh(np.dot(self.cell_weight,z) + self.cell_bias)
        return [it,ct]

    def cell_gate(self,ft,it,ct_new,ct1):
        return ft*ct1 + it*ct_new

    def output_gate(self,z,ct):
        ot = af.sigmoid(np.dot(self.output_weight,z) + self.output_bias)
        return ot*af.tanh(ct)