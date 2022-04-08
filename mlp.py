import torch as t
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from keras.datasets.mnist import load_data

class Mlp(nn.Module):
    def __init__(self, input_dim = 28*28, hidden_dim = 1200, output_dim = 10):
        super(Mlp, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)

