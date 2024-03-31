"""
The Categorical DQN algorithm.

references:
- cleanrl: https://docs.cleanrl.dev/rl-algorithms/c51/
- cleanrl codes (categorical dqn): https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51.py
- original papers:
    * https://proceedings.mlr.press/v70/bellemare17a.html?trk=public_post_comment-text

! Note: the code is completed with the help of copilot.
"""

import gymnasium as gym

import numpy as np

import torch
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.buffers import ReplayBuffer

import os
import random
import datetime
import time

# * The Categorical DQN is based on the DQN algorithm
from RLAlgos.DQN import DQN


class CategoricalDQN(DQN):
    def __init__(self, env, q_network_class, exp_name="categorical-dqn", seed=1, cuda=0, n_atoms=51, v_min=-10,
                 v_max=10, learning_rate=2.5e-4, buffer_size=10000, rb_optimize_memory=False, gamma=0.99, tau=1.,
                 target_network_frequency=500, batch_size=128, start_e=1, end_e=0.05, exploration_fraction=0.5,
                 train_frequency=10, write_frequency=100, save_folder="./categorical-dqn/"):
        """
        The Categorical DQN algorithm.
        :param env: The environment to train the agent on
        :param q_network_class: The class of the Q network to use
        :param exp_name: The name of the experiment
        :param seed: The seed for the random number generators
        :param cuda: The cuda device to use
        :param n_atoms: The number of atoms in the distribution
        :param v_min: The minimum value of the distribution
        :param v_max: The maximum value of the distribution
        :param learning_rate: The learning rate of the optimizer
        :param buffer_size: The size of the replay buffer
        :param rb_optimize_memory: Whether to optimize the memory of the replay buffer
        :param gamma: The discount factor
        :param target_network_frequency: The frequency of updating the target network
        :param batch_size: The batch size
        :param start_e: The starting value of epsilon
        :param end_e: The ending value of epsilon
        :param exploration_fraction: The fraction of the total number of steps over which the epsilon is annealed
        :param train_frequency: The frequency of training
        :param write_frequency: The frequency of writing to tensorboard
        :param save_folder: The folder to save the model
        """

        super(CategoricalDQN, self).__init__(env, q_network_class, exp_name, seed, cuda, learning_rate, buffer_size,
                                             rb_optimize_memory, gamma, tau, target_network_frequency, batch_size,
                                             start_e, end_e, exploration_fraction, train_frequency, write_frequency,
                                             save_folder)

        # the networks
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.q_network = q_network_class(self.env, n_atoms=n_atoms, v_min=v_min, v_max=v_max).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate, eps=0.01 / batch_size)
        self.target_network = q_network_class(self.env, n_atoms=n_atoms, v_min=v_min, v_max=v_max).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

    def learn(self, total_timesteps=500000, learning_starts=10000):

        # start the game
        obs, _ = self.env.reset(seed=self.seed)
        for global_step in range(total_timesteps):

            epsilon = self.linear_schedule(self.exploration_fraction * total_timesteps, global_step)

            if random.random() < epsilon:
                action = self.env.action_space.sample()
            else:
                actions, pmf = self.q_network.get_action(torch.Tensor(np.expand_dims(obs, axis=0)).to(self.device))
                action = actions.cpu().numpy()

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            if "episode" in info:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                self.writer.add_scalar("charts/epsilon", epsilon, global_step)

            self.replay_buffer.add(obs, next_obs, action, reward, done, info)

            if not done:
                obs = next_obs
            else:
                obs, _ = self.env.reset()

            if global_step > learning_starts:
                if global_step % self.train_frequency == 0:
                    self.optimize(global_step)

        self.env.close()
        self.writer.close()

    def optimize(self, global_step):
        data = self.replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            _, next_pmfs = self.target_network.get_action(data.next_observations)
            next_atoms = data.rewards + self.gamma * (1 - data.dones) * self.target_network.atoms

            # prediction
            delta_z = self.target_network.atoms[1] - self.target_network.atoms[0]
            tz = next_atoms.clamp(self.v_min, self.v_max)

            b = (tz - self.v_min) / delta_z
            l = b.floor().clamp(0, self.n_atoms - 1)
            u = b.ceil().clamp(0, self.n_atoms - 1)
            d_m_l = (u + (l == u).float() - b) * next_pmfs
            d_m_u = (b - l) * next_pmfs
            target_pmfs = torch.zeros_like(next_pmfs)
            for i in range(target_pmfs.size(0)):
                target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

        _, old_pmfs = self.q_network.get_action(data.observations, data.actions.flatten())
        loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

        if global_step % self.write_frequency == 0:
            self.writer.add_scalar("losses/loss", loss.item(), global_step)
            old_val = (old_pmfs * self.q_network.atoms).sum(1)
            self.writer.add_histogram("charts/old_val", old_val.mean().item(), global_step)

        # * update q network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # * update target network
        if global_step % self.target_network_frequency == 0:
            for target_network_param, q_network_param in zip(self.target_network.parameters(),
                                                             self.q_network.parameters()):
                target_network_param.data.copy_(
                    self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data)
