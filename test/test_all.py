import pytest as pt
from neuron import neuron

def test_neuron_init():
    new_neuron = neuron(1,1)
    assert sum(new_neuron.cell_bias) != 0

