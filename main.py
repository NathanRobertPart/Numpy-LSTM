import models
import numpy as np
import lstm_layer as lstm

def train_time_steps(x,y_true,h_prev,ct_prev,dimensionality,h_size,z_size,lstm_instance):
    timeStep = models.TimeStepModel()
    lstm_instance = lstm.lstm_layer(h_size,z_size)

    timeStep.ht_s[-1] = np.copy(h_prev)
    timeStep.ct_s[-1] = np.copy(ct_prev)

    loss = 0
    for i in range(len(x)):
        timeStep.x_s[i] = np.zeros((dimensionality, 1))
        timeStep.x_s[i][y_true[i]] = 1
        timeStep = lstm.passes.forward_pass(timeStep,timeStep.x_s[i],timeStep.h_s[i-1],timeStep.ct_s[i-1],lstm_instance,True)

        loss += -np.log(timeStep.y_s[i][y_true[i], 0])
        print(loss)

    lstm_instance = lstm.passes.reset_grad(lstm_instance)

    dht_next = np.zeros_like(timeStep.ht_s[0])
    dct_next = np.zeros_like(timeStep.ct_s[0])

    x_reverse = x.reverse()

    for i in range(len(x_reverse)):
        dht_next, dct_next,timeStep = lstm.passes.backward_pass(timeStep,i,y_true,dht_next,dct_next,lstm_instance,True,h_size)

    lstm_instance = lstm.passes.clip_grad(lstm_instance)

    return loss, timeStep.ht_s[len(x) - 1], timeStep.ct_s[len(x) - 1],lstm_instance