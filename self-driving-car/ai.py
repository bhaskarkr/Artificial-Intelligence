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


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()

    def push(self, event):
        # event = last state, new state, last action, last reward
        self.memory.append(event)
        if len(self.memory > self.capacity):
            del self.memory[0]

    def sample(self, batch_size):
        """
            LIST [[1,2,3], [4,5,6]] then zip(*LIST) = [[1,4], [2,5], [3,6]]
            So if A and B are of shape (3, 4):
            torch.cat([A, B], dim=0) will be of shape (6, 4)
            torch.stack([A, B], dim=0) will be of shape (2, 3, 4)
        """
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
