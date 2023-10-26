"""
The Proximal Policy Optimization (PPO) algorithm.

* Both discrete and continuous action spaces are supported.

references:
- cleanrl: https://docs.cleanrl.dev/rl-algorithms/ppo/
- cleanrl codes (ppo classic control): https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
- cleanrl codes (ppo atari): https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py
- cleanrl codes (ppo continuous): https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
- original papers:
    * https://arxiv.org/abs/1707.06347
    * https://arxiv.org/abs/2005.12729
    * https://arxiv.org/abs/1707.02286

! Note: the code is completed with the help of copilot.
"""

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.buffers import ReplayBuffer

import os
import random
import datetime
import time


class PPO:
    """
    The Proximal Policy Optimization (PPO) algorithm.
    """

    def __init__(self, env, agent_class, exp_name="ppo", seed=1, cuda=0, gamma=0.99, gae_lambda=0.95,
                 rollout_length=2048, lr=2.5e-4, eps=1e-5, anneal_lr=True, num_minibatches=4, update_epochs=4,
                 clip_coef=0.2, clip_value_loss=True, entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5,
                 target_kl=None, write_frequency=100, save_folder="./ppo/"):
        """
        The initialization of the PPO class.
        :param env: the gymnasium-based environment.
        :param agent_class: the class of the agent.
        :param exp_name: the name of the experiment.
        :param seed: the random seed.
        :param cuda: the cuda device.
        :param gamma: the discount factor.
        :param gae_lambda: the lambda coefficient in generalized advantage estimation.
        :param rollout_length: the rollout length.
        :param lr: the learning rate.
        :param eps: the epsilon value.
        :param anneal_lr: whether to anneal the learning rate.
        :param num_minibatches: the number of minibatches.
        :param update_epochs: the number of update epochs.
        :param clip_coef: the clipping coefficient.
        :param clip_value_loss: whether to clip the value loss.
        :param entropy_coef: the entropy coefficient.
        :param value_coef: the value coefficient.
        :param max_grad_norm: the maximum gradient norm.
        :param target_kl: the target kl divergence.
        :param write_frequency: the frequency of writing logs.
        :param save_folder: the folder to save the model.
        """

        self.exp_name = exp_name

        self.seed = seed

        # set the random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

        self.env = env

        self.agent = agent_class(self.env).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr, eps=eps)

        # set up the storage
        self.obs = torch.zeros((rollout_length,) + self.env.observation_space.shape).to(self.device)
