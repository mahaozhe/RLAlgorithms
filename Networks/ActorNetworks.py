"""
The script to define the actor networks.

The input is usually the observation.
The output is usually the action or the distribution parameters.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from utils.network_utils import layer_init_bias


class SACActor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
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


class SACActorAtari(nn.Module):
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
        self.fc_logits = layer_init_bias(nn.Linear(512, env.action_space.n))

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)

        return logits

    def get_action(self, x):
        logits = self(x / 255.0)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


class SACActorMiniGrid(nn.Module):
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
            layer_init_bias(nn.Linear(n_flatten, 120)),
            nn.ReLU(),
            layer_init_bias(nn.Linear(120, 84)),
            nn.ReLU(),
            layer_init_bias(nn.Linear(84, env.action_space.n)),
        )

    def forward(self, x):
        cnn_features = self.cnn(x)
        logits = self.network(cnn_features)
        return logits

    def get_action(self, x):
        logits = self.forward(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


class DeterministicActorContinuousControl(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias
