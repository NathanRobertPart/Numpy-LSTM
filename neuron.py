import numpy as np


class neuron(object):

    def __init__(self,H_size,z_size):
        self.forget_weights = np.zeros([H_size,z_size])
        self.input_weight = np.zeros([H_size,z_size])
        self.cell_weight = np.zeros([H_size,z_size])
        self.output_weight = np.zeros([H_size,z_size])

        self.forget_bias = np.zeros([H_size,1])
        self.input_bias = np.zeros([H_size, 1])
        self.cell_bias = np.zeros([H_size, 1])
        self.output_bias = np.zeros([H_size, 1])

