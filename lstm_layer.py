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
    def forward_pass(timeStepModel,t,x,h_prev,ct_prev,neuron,final):
        z = np.row_stack((h_prev, x))
        ht, ot, z, ft, it, ct_new, ct = neuron.cell_state(z, ct_prev)
        if final:
            y,vt = neuron.final_output(ht)
        else:
            y,vt = 0,0

        timeStepModel.y_s[t] = y
        timeStepModel.zt_s[t] = z
        timeStepModel.ht_s[t] = ht
        timeStepModel.ot_s[t] = ot
        timeStepModel.ft_s[t] = ft
        timeStepModel.it_s[t] = it
        timeStepModel.ct_new_s[t] = ct_new
        timeStepModel.ct_s[t] = ct

        return timeStepModel

    @staticmethod
    def backward_pass(timeStepModel,t,y_true,dh_next,dct_next,neuron,final,h_size):

        ht = timeStepModel.ht_s[t]
        y = timeStepModel.y_s[t]
        ot = timeStepModel.ot_s[t]
        it = timeStepModel.it_s[t]
        z = timeStepModel.zt_s[t]
        ct = timeStepModel.ct_s[t]
        ct_new = timeStepModel.ct_new_s[t]
        ct_prev = timeStepModel.ct_s[t-1]
        ft = timeStepModel.ft_s[t]

        neuron,dv = gc.softmaxPropogation(neuron,ht,y_true,y,final)
        neuron,dh,do = gc.outputPropogation(neuron, dv, dh_next, ot, z, ct)
        neuron,dct,dct_new = gc.cellPropogation(neuron, dct_next, dh, ot, it, ct, ct_new, z)
        neuron,dit = gc.inputPropogation(neuron, dct, ct_new, it, z)
        neuron,df = gc.forgetPropogation(neuron,dct,ct_prev,ft,z)
        dz, dh_prev, dct_prev = gc.aggregation(neuron, df, dit, dct_new, do, h_size, ft, dct)
        return [timeStepModel,dh_prev,dct_prev]


    @staticmethod
    def reset_grad(lstm_layer):
        for item in lstm_layer.all():
            item.grad.fill(0)
        return lstm_layer

    @staticmethod
    def clip_grad(lstm_layer):
        for item in lstm_layer.all():
            np.clip(item.grad, -1, 1, out=item.grad)
        return lstm_layer
