import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


class Network(nn.Module):

    def __init__(self, number_input_neuron, number_output_neuron_actions):
        super(Network, self).__init__()
        self.number_input_neuron = number_input_neuron
        self.number_output_neuron_actions = number_output_neuron_actions
        number_of_neuron_in_hidden_layer = 30
        self.fullConnection1 = nn.Linear(number_input_neuron, number_of_neuron_in_hidden_layer)
        self.fullConnection2 = nn.Linear(number_of_neuron_in_hidden_layer, number_output_neuron_actions)

    def forward(self, state):
        x = F.relu(self.fullConnection1(state))
        q_values = self.fullConnection2(x)
        return q_values
