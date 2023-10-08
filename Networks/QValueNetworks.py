"""
The script to define the (Q) value networks.

The input is usually the observation (with the action sometimes).
The output is usually the value of the observation or Q-value of the observation-action pair.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class SACSoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x