"""
The Deep Q Network (DQN) algorithm.

references:
- cleanrl: https://docs.cleanrl.dev/rl-algorithms/dqn/
- cleanrl codes (dqn): https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py
- cleanrl codes (dqn atari): https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py
- original papers:
    * https://www.nature.com/articles/nature14236
    * https://arxiv.org/pdf/1312.5602.pdf

! Note: the code is completed with the help of copilot.
"""

import gymnasium as gym

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


class DQN:
    """
    The DQN base algorithm.
    """

    def __init__(self, env, q_network_class, exp_name="dqn", seed=1, cuda=0, learning_rate=2.5e-4, buffer_size=10000,
                 rb_optimize_memory=False, gamma=0.99, tau=1., target_network_frequency=500, batch_size=128, start_e=1,
                 end_e=0.05, exploration_fraction=0.5, train_frequency=10, write_frequency=100, save_folder="./dqn/"):
        """
        Initialize the DQN algorithm.
        :param env: the gymnasium-based environment
        :param q_network_class: the agent class
        :param exp_name: the experiment name
        :param seed: the random seed
        :param cuda: whether to use cuda
        :param learning_rate: the learning rate
        :param buffer_size: the replay memory buffer size
        :param rb_optimize_memory: whether to optimize the memory usage of the replay buffer
        :param gamma: the discount factor gamma
        :param tau: the target network update rate
        :param target_network_frequency: the timesteps it takes to update the target network
        :param batch_size: the batch size of sample from the reply memory
        :param start_e: the starting epsilon for exploration
        :param end_e: the ending epsilon for exploration
        :param exploration_fraction: the fraction of `total-timesteps` it takes from start-e to go end-e
        :param train_frequency: the frequency of training
        :param write_frequency: the frequency of writing to tensorboard
        :param save_folder: the folder to save the model
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

        assert isinstance(self.env.action_space, gym.spaces.Discrete), "only discrete action space is supported for DQN"

        # the networks
        self.q_network = q_network_class(self.env).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.target_network = q_network_class(self.env).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # the replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            optimize_memory_usage=rb_optimize_memory,
            handle_timeout_termination=False
        )

        self.gamma = gamma

        # for the epsilon greedy exploration
        self.start_e = start_e
        self.end_e = end_e
        self.exploration_fraction = exploration_fraction

        # for the batch training
        self.batch_size = batch_size

        # for the target network update
        self.target_network_frequency = target_network_frequency
        self.tau = tau

        # for the training
        self.train_frequency = train_frequency

        # * for the tensorboard writer
        run_name = "{}-{}-{}-{}".format(exp_name, env.unwrapped.spec.id, seed,
                                        datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S'))
        os.makedirs("./runs/", exist_ok=True)
        self.writer = SummaryWriter(os.path.join("./runs/", run_name))
        self.write_frequency = write_frequency

        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)

    def linear_schedule(self, duration, t):
        """
        Linear interpolation between start_e and end_e
        :param duration: the fraction of `total-timesteps` it takes from start-e to go end-e
        :param t: the current timestep
        """
        slope = (self.end_e - self.start_e) / duration
        return max(slope * t + self.start_e, self.end_e)

    def learn(self, total_timesteps=500000, learning_starts=10000):

        # start the game
        obs, _ = self.env.reset(seed=self.seed)
        for global_step in range(total_timesteps):

            epsilon = self.linear_schedule(self.exploration_fraction * total_timesteps, global_step)

            if random.random() < epsilon:
                action = self.env.action_space.sample()
            else:
                q_value = self.q_network(torch.Tensor(np.expand_dims(obs, axis=0)).to(self.device))
                action = torch.argmax(q_value, dim=1).cpu().numpy()

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
            target_max, _ = self.target_network(data.next_observations).max(dim=1)
            td_target = data.rewards.flatten() + self.gamma * target_max * (1 - data.dones.flatten())
        old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
        loss = F.mse_loss(td_target, old_val)

        if global_step % self.write_frequency == 0:
            self.writer.add_scalar("losses/td_loss", loss, global_step)
            self.writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)

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

    def save(self, indicator="best"):
        if indicator.startswith("best") or indicator.startswith("final"):
            torch.save(self.target_network.state_dict(),
                       os.path.join(self.save_folder,
                                    "q_network-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed)))
        else:
            torch.save(self.target_network.state_dict(),
                       os.path.join(self.save_folder,
                                    "/q_network-{}-{}-{}-{}.pth".format(self.exp_name,
                                                                        indicator,
                                                                        self.seed,
                                                                        datetime.datetime.fromtimestamp(
                                                                            time.time()).strftime(
                                                                            '%Y-%m-%d-%H-%M-%S'))))


class NoisyNetDQN(DQN):
    """
    The NoisyNet DQN algorithm.
    """

    def __init__(self, env, noisy_q_network_class, exp_name="noisy-net-dqn", seed=1, cuda=0, learning_rate=2.5e-4,
                 buffer_size=10000, rb_optimize_memory=False, gamma=0.99, tau=1., target_network_frequency=500,
                 batch_size=128, train_frequency=10, noisy_std_init=0.25, write_frequency=100,
                 save_folder="./noisy-net-dqn/"):
        """
        Initialize the NoisyNet DQN algorithm.
        :param env: the gym-based environment
        :param noisy_q_network_class: the noisy network class
        :param exp_name: the experiment name
        :param seed: the random seed
        :param cuda: whether to use cuda
        :param learning_rate: the learning rate
        :param buffer_size: the replay memory buffer size
        :param rb_optimize_memory: whether to optimize the memory usage of the replay buffer
        :param gamma: the discount factor gamma
        :param tau: the target network update rate
        :param target_network_frequency: the timesteps it takes to update the target network
        :param batch_size: the batch size of sample from the reply memory
        :param train_frequency: the frequency of training
        :param noisy_std_init: the initial standard deviation for the noisy network
        :param write_frequency: the frequency of writing to tensorboard
        :param save_folder: the folder to save the model
        """

        super(NoisyNetDQN, self).__init__(env=env, q_network_class=noisy_q_network_class, exp_name=exp_name, seed=seed,
                                          cuda=cuda, learning_rate=learning_rate, buffer_size=buffer_size,
                                          rb_optimize_memory=rb_optimize_memory, gamma=gamma, tau=tau,
                                          target_network_frequency=target_network_frequency, batch_size=batch_size,
                                          train_frequency=train_frequency, write_frequency=write_frequency,
                                          save_folder=save_folder)

        # the noisy networks
        self.q_network = noisy_q_network_class(self.env, std_init=noisy_std_init).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.target_network = noisy_q_network_class(self.env, std_init=noisy_std_init).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

    def learn(self, total_timesteps=500000, learning_starts=10000):
        # start the game
        obs, _ = self.env.reset(seed=self.seed)
        for global_step in range(total_timesteps):
            # there is no need for epsilon-greedy exploration in NoisyNet
            q_value = self.q_network(torch.Tensor(np.expand_dims(obs, axis=0)).to(self.device))
            action = torch.argmax(q_value, dim=1).cpu().numpy()[0]

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            if "episode" in info:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

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

    # def optimize_noisy_net(self, global_step):
    #     # reset the noise
    #     self.q_network.reset_noise()
    #
    #     self.optimize(global_step)
