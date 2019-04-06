import pytest as pt
from lstm_layer import lstm_layer
from load_data import load_data

def test_neuron_init():
    new_neuron = lstm_layer(1,1)
    assert sum(new_neuron.cell_bias.values) != 0

def test_load_data():
    load_data()

