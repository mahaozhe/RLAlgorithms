"""
The network classes for NoisyNet.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class NoisyQNetClassicControl(nn.Module):
    """
    The Noisy version Q network for the classic control environment.
    """

    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            NoisyLinear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            NoisyLinear(120, 84),
            nn.ReLU(),
            NoisyLinear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)

    def reset_noise(self):
        for layer in self.network:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


class NoisyQNetMiniGrid(nn.Module):
    """
    The Noisy version Q network for the minigrid environment.
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
            NoisyLinear(n_flatten, 120),
            nn.ReLU(),
            NoisyLinear(120, 84),
            nn.ReLU(),
            NoisyLinear(84, env.action_space.n),
        )

    def forward(self, x):
        cnn_features = self.cnn(x)
        return self.network(cnn_features)

    def reset_noise(self):
        """
        Reset the noise for all NoisyLinear layers in the network.
        This should be called at the start of each update/episode.
        """
        for layer in self.network:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
