"""
The scrit to define some Q-Networks.

The input is usually the observation.
The output is usually the Q value for each action.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.network_utils import layer_init_bias


class QNetClassicControl(nn.Module):
    """
    The Q network for classic control environments.
    The observation space is usually a vector.
    The action space is a discrete vector.
    """

    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)


class QNetMiniGrid(nn.Module):
    """
    The Q network for the minigrid environment.
    The observation space is usually a matrix.
    The action space is a discrete vector.

    The structure is referred to the MiniGrid Documentation.
    """

    def __init__(self, env):
        super().__init__()
        # Assume observation_space is a gym Space with shape (channels, height, width)
        n_input_channels = env.observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flat features size by doing a forward pass through cnn with a dummy input
        with torch.no_grad():
            dummy_input = torch.as_tensor(env.observation_space.sample()[None]).float()
            n_flatten = self.cnn(dummy_input).shape[1]

        self.network = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n),
        )

    def forward(self, x):
        cnn_features = self.cnn(x)
        return self.network(cnn_features)


class QNetAtari(nn.Module):
    """
    The Q network for Atari environments.
    The observation space is usually an image.
    The action space is a discrete vector.
    """

    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


class SACSoftQNetworkAtari(nn.Module):
    """
    The soft Q-network for Atari games.
    """

    def __init__(self, env):
        super().__init__()
        obs_shape = env.observation_space.shape

        self.conv = nn.Sequential(
            layer_init_bias(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init_bias(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init_bias(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init_bias(nn.Linear(output_dim, 512))
        self.fc_q = layer_init_bias(nn.Linear(512, env.action_space.n))

    def forward(self, x):
        x = F.relu(self.conv(x / 255.0))
        x = F.relu(self.fc1(x))
        q_vals = self.fc_q(x)
        return q_vals

class SACSoftQNetworkMiniGrid(nn.Module):
    """
    The soft Q-network adapted for MiniGrid environments.
    """

    def __init__(self, env):
        super().__init__()
        n_input_channels = env.observation_space.shape[0]

        self.cnn = nn.Sequential(
            layer_init_bias(nn.Conv2d(n_input_channels, 16, (2, 2))),
            nn.ReLU(),
            layer_init_bias(nn.Conv2d(16, 32, (2, 2))),
            nn.ReLU(),
            layer_init_bias(nn.Conv2d(32, 64, (2, 2))),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flat features size by doing a forward pass through cnn with a dummy input
        with torch.no_grad():
            dummy_input = torch.as_tensor(env.observation_space.sample()[None]).float()
            n_flatten = self.cnn(dummy_input).shape[1]

        self.network = nn.Sequential(
            layer_init_bias(nn.Linear(n_flatten, 128)),
            nn.ReLU(),
            layer_init_bias(nn.Linear(128, env.action_space.n)),
        )

    def forward(self, x):
        cnn_features = self.cnn(x)
        q_vals = self.network(cnn_features)
        return q_vals