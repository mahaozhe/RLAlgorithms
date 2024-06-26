"""
The network classes for NoisyNet.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.25):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        self.weight_epsilon.copy_(torch.ger(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        if self.training:
            self.reset_noise()
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

    def __init__(self, env, std_init=1):
        super().__init__()
        self.network = nn.Sequential(
            NoisyLinear(np.array(env.observation_space.shape).prod(), 120, std_init=std_init),
            nn.ReLU(),
            NoisyLinear(120, 84, std_init=std_init),
            nn.ReLU(),
            NoisyLinear(84, env.action_space.n, std_init=std_init),
        )

    def forward(self, x):
        return self.network(x)

    # def reset_noise(self):
    #     for layer in self.network:
    #         if isinstance(layer, NoisyLinear):
    #             layer.reset_noise()


class NoisyQNetMiniGrid(nn.Module):
    """
    The Noisy version Q network for the minigrid environment.
    The observation space is usually a matrix.
    The action space is a discrete vector.

    The structure is referred to the MiniGrid Documentation.
    """

    def __init__(self, env, std_init=1):
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
            NoisyLinear(n_flatten, 128, std_init=std_init),
            nn.ReLU(),
            NoisyLinear(128, env.action_space.n, std_init=std_init),
        )

    def forward(self, x):
        cnn_features = self.cnn(x)
        return self.network(cnn_features)

    # def reset_noise(self):
    #     """
    #     Reset the noise for all NoisyLinear layers in the network.
    #     This should be called at the start of each update/episode.
    #     """
    #     for layer in self.network:
    #         if isinstance(layer, NoisyLinear):
    #             layer.reset_noise()


class NoisyQNetworkContinuousControl(nn.Module):
    def __init__(self, env, std_init=10):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256)
        self.fc2 = NoisyLinear(256, 256, std_init=std_init)
        self.fc3 = NoisyLinear(256, 1, std_init=std_init)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NoisySACActor(nn.Module):
    def __init__(self, env, std_init=10):
        super().__init__()
        self.fc1 = NoisyLinear(np.array(env.observation_space.shape).prod(), 256, std_init=std_init)
        self.fc2 = NoisyLinear(256, 256, std_init=std_init)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.log_std_max = 2
        self.log_std_min = -5

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
