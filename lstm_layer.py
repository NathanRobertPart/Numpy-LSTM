import numpy as np
from activation_functions import activation_functions as af
from activation_functions import activation_function_derivations as df

class matrixModel(object):

    def __init__(self,H_size,z_size):
        self.values = np.random.randn(H_size,z_size)
        self.grad = np.zeros([H_size,z_size])

class lstm_layer(object):

    def __init__(self,H_size,z_size):
        self.initialise_weights(H_size,z_size)

    def initialise_weights(self,H_size,z_size):
        self.forget_weight = matrixModel(H_size,z_size)
        self.input_weight = matrixModel(H_size,z_size)
        self.cell_weight = matrixModel(H_size,z_size)
        self.output_weight = matrixModel(H_size,z_size)
        self.final_weight = matrixModel(H_size,z_size)

        self.forget_bias = matrixModel(H_size,1)
        self.input_bias = matrixModel(H_size, 1)
        self.cell_bias = matrixModel(H_size, 1)
        self.output_bias = matrixModel(H_size, 1)
        self.final_bias = matrixModel(H_size, 1)

    def forget_gate(self,z):
        return af.sigmoid(np.dot(self.forget_weight.values,z) + self.forget_bias.values)

    def input_gate(self,z):
        it = af.sigmoid(np.dot(self.input_weight.values,z) + self.input_bias.values)
        ct_new = af.tanh(np.dot(self.cell_weight.values,z) + self.cell_bias.values)
        return [it,ct_new]

    def cell_gate(self,ft,it,ct_new,ct1):
        return ft*ct1 + it*ct_new

    def output_gate(self,z,ct):
        ot = af.sigmoid(np.dot(self.output_weight.values,z) + self.output_bias.values)
        return ot*af.tanh(ct),ot

    def cell_state(self,z,ct1):
        ft = self.forget_gate(z)
        it,ct_new = self.input_gate(z)
        ct = self.cell_gate(ft,it,ct_new,ct1)
        ht,ot = self.output_gate(z,ct)
        return ht,ot,z,ft,it,ct_new,ct

    def final_output(self,ht):
        vt = np.dot(self.final_weight.values,ht) + self.final_bias.values
        y = np.exp(vt) / np.sum(np.exp(vt))
        return y,vt



class passes():

    @staticmethod
    def forward_pass(x,h_prev,Ct_prev,neuron,final):
        z = np.row_stack((h_prev, x))
        ht, ot, z, ft, it, ct_new, ct = neuron.cell_state(z, Ct_prev)
        if final:
            y,vt = neuron.final_output(ht)
        else:
            y,vt = 0,0
        return y, vt, ht, ot, z, ft, it, ct_new, ct

    @staticmethod
    def backward_pass(y_true,dh_next,dC_next,Ct_prev,y,vt,ht,ot,z,ft,it,neuron,final):

        dv = np.copy(y)
        dv = [dv[a] - 1 for a in y_true]

        if final:
            neuron.final_weight.grad += np.dot(dv, ht)
            neuron.final_bias.grad += dv

        dh = np.dot(neuron.final_weight.values.T, dv)

