import numpy as np
import activation_functions as af
import gradient_calculations as gc

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
    def backward_pass(y_true,dh_next,dct_next,ct_prev,y,vt,ht,ot,z,ft,it,ct_new, ct,neuron,final,h_size):

        dv = np.copy(y)
        dv = [dv[a] - 1 for a in y_true]

        if final:
            neuron.final_weight.grad += np.dot(dv, ht)
            neuron.final_bias.grad += dv

        neuron,dh,do = gc.outputPropogation(neuron, dv, dh_next, ot, z, ct)
        neuron,dct,dct_new = gc.cellPropogation(neuron, dct_next, dh, ot, it, ct, ct_new, z)
        neuron,dit = gc.inputPropogation(neuron, dct, ct_new, it, z)
        neuron,df = gc.forgetPropogation(neuron,dct,ct_prev,ft,z)
        dz, dh_prev, dct_prev = gc.aggregation(neuron, df, dit, dct_new, do, h_size, ft, dct)
        return [dh_prev,dct_prev]


