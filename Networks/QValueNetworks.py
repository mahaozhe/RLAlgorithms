"""
The script to define the (Q) value networks.

The input is usually the observation (with the action sometimes).
The output is usually the value of the observation or Q-value of the observation-action pair.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetworkContinuousControl(nn.Module):
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


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.ReLU()):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.activation = activation

    def forward(self, x):
        residual = x
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x += residual
        x = self.activation(x)
        return x


class QNetworkContinuousControlNew(nn.Module):
    def __init__(self, env, block_num=3):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256)
        self.hidden_blocks = nn.ModuleList([ResidualBlock(256, 256) for _ in range(block_num)])
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.fc1(x)
        for block in self.hidden_blocks:
            x = block(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CategoricalQNetwork(nn.Module):
    def __init__(self, env, n_atoms=51, v_min=-10, v_max=10):
        super().__init__()
        self.n_atoms = n_atoms
        self.n = env.action_space.n
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.n * n_atoms)
        )

    def get_action(self, x, action=None):
        logits = self.network(x)
        # probability mass function
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (self.atoms * pmfs).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)), action]
