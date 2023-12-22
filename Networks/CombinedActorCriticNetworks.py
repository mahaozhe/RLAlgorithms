"""
For the actor-critic networks with shared feature extractors.

The input is usually the observation (with the action).
The output is divided into two branches: one for action and one for (Q-)value.
"""

import gymnasium as gym

import numpy as np

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from utils.network_utils import layer_init_std_bias


class PPOClassicControlAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        assert isinstance(envs, gym.vector.SyncVectorEnv), "only SyncVectorEnv is supported!"

        self.critic = nn.Sequential(
            layer_init_std_bias(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init_std_bias(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init_std_bias(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init_std_bias(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init_std_bias(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init_std_bias(nn.Linear(64, envs.single_action_space.shape), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class PPOAtariAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        self.network = nn.Sequential(
            layer_init_std_bias(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init_std_bias(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init_std_bias(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init_std_bias(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init_std_bias(nn.Linear(512, envs.single_action_space.shape), std=0.01)
        self.critic = layer_init_std_bias(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class PPOMujocoAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        self.critic = nn.Sequential(
            layer_init_std_bias(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init_std_bias(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init_std_bias(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init_std_bias(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init_std_bias(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init_std_bias(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class PPORNDAgent(nn.Module):
    """agent network"""

    def __init__(self, envs):
        super().__init__()
        # --- original atari model
        # self.network = nn.Sequential(
        #     layer_init(nn.Conv2d(4, 32, 8, stride=4)),
        #     nn.ReLU(),
        #     layer_init(nn.Conv2d(32, 64, 4, stride=2)),
        #     nn.ReLU(),
        #     layer_init(nn.Conv2d(64, 64, 3, stride=1)),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     layer_init(nn.Linear(64 * 7 * 7, 256)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(256, 448)),
        #     nn.ReLU(),
        # )
        # self.extra_layer = nn.Sequential(layer_init(nn.Linear(448, 448), std=0.1), nn.ReLU())
        # self.actor = nn.Sequential(
        #     layer_init(nn.Linear(448, 448), std=0.01),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(448, envs.single_action_space.shape[0]), std=0.01),
        # )
        # self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
        # self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)

        self.actor_mean = nn.Sequential(
            layer_init_std_bias(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init_std_bias(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init_std_bias(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        self.critic_ext = nn.Sequential(
            layer_init_std_bias(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init_std_bias(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init_std_bias(nn.Linear(64, 1), std=1.0),
        )
        self.critic_int = nn.Sequential(
            layer_init_std_bias(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init_std_bias(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init_std_bias(nn.Linear(64, 1), std=1.0),
        )

    def get_action_and_value(self, x, action=None):
        # hidden = self.network(x / 255.0)
        # logits = self.actor(hidden)
        # probs = Categorical(logits=logits)
        # features = self.extra_layer(hidden)

        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            # probs.log_prob(action),
            # probs.entropy(),
            # self.critic_ext(features + hidden),
            # self.critic_int(features + hidden),
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic_ext(x),
            self.critic_int(x),
        )

    def get_value(self, x):
        # hidden = self.network(x / 255.0)
        # features = self.extra_layer(hidden)
        # return self.critic_ext(features + hidden), self.critic_int(features + hidden)
        return self.critic_ext(x), self.critic_int(x)


class RPOMujocoAgent(nn.Module):
    def __init__(self, envs, rpo_alpha, cuda=0):
        super().__init__()

        self.rpo_alpha = rpo_alpha
        self.device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
        self.critic = nn.Sequential(
            layer_init_std_bias(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init_std_bias(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init_std_bias(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init_std_bias(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init_std_bias(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init_std_bias(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        # + new to RPO
        else:
            # sample again to add stochasticity to the policy
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(self.device)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RNDModel(nn.Module):
    def __init__(self, envs):
        super().__init__()

        self.envs = envs

        # self.input_size = input_size
        # self.output_size = output_size
        # feature_output = 7 * 7 * 64
        # Prediction network
        # self.predictor = nn.Sequential(
        #     layer_init(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)),
        #     nn.LeakyReLU(),
        #     layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
        #     nn.LeakyReLU(),
        #     layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
        #     nn.LeakyReLU(),
        #     nn.Flatten(),
        #     layer_init(nn.Linear(feature_output, 512)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(512, 512)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(512, 512)),
        # )

        # Target network
        # self.target = nn.Sequential(
        #     layer_init(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)),
        #     nn.LeakyReLU(),
        #     layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
        #     nn.LeakyReLU(),
        #     layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
        #     nn.LeakyReLU(),
        #     nn.Flatten(),
        #     layer_init(nn.Linear(feature_output, 512)),
        # )

        self.predictor = nn.Sequential(
            layer_init_std_bias(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init_std_bias(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init_std_bias(nn.Linear(64, 1), std=1.0),
        )
        self.target = nn.Sequential(
            layer_init_std_bias(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init_std_bias(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init_std_bias(nn.Linear(64, 1), std=1.0),
        )
        # target network is not trainable
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature
