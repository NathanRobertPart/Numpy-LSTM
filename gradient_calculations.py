import numpy as np
import activation_functions as af

def softmaxPropogation(neuron,ht,y_true,y,final):
    dv = np.copy(y)
    dv = [dv[a] - 1 for a in y_true]
    if final:
        neuron.final_weight.grad += np.dot(dv, ht)
        neuron.final_bias.grad += dv
    return neuron,dv

def outputPropogation(neuron,dv,dh_next,ot,z,ct):
    dh = np.dot(neuron.final_weight.values.T, dv) + dh_next
    do = dh * af.tanh(ct) * af.dsigmoid(ot)
    neuron.output_weight.grad += np.dot(do, z.T)
    neuron.output_bias.grad += do
    return neuron,dh,do

def cellPropogation(neuron,dct_next,dh,ot,it,ct,ct_new,z):
    dct = np.copy(dct_next)
    dct += dh * ot * af.dtanh(af.tanh(ct))
    dct_new = dct * it * af.dtanh(ct_new)
    neuron.cell_weight.values += np.dot(dct_new, z.T)
    neuron.cell_weight.bias += dct_new
    return neuron,dct,dct_new

def inputPropogation(neuron,dct,ct_new,it,z):
    dit = dct * ct_new * af.dsigmoid(it)
    neuron.input_weight.values += np.dot(dit, z.T)
    neuron.input_bias.values += dit
    return neuron,dit

def forgetPropogation(neuron,dct,ct_prev,ft,z):
    df = dct * ct_prev * af.dsigmoid(ft)
    neuron.forget_weight.values += np.dot(df, z.T)
    neuron.forget_bias.values += df
    return neuron,df

def aggregation(neuron,df,dit,dct_new,do,h_size,ft,dct):
    dz = (np.dot(neuron.forget_weight.values.T, df)
          + np.dot(neuron.input_weight.values.T, dit)
          + np.dot(neuron.cell_weight.values.T, dct_new)
          + np.dot(neuron.output_weight.values.T, do))
    dh_prev = dz[:h_size, :]
    dct_prev = ft * dct
    return dz,dh_prev,dct_prev