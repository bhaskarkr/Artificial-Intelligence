
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

