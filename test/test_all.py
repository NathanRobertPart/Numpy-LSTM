import pytest as pt
from lstm_layer import lstm_layer

def test_neuron_init():
    new_neuron = lstm_layer(1,1)
    assert sum(new_neuron.cell_bias.values) != 0

