"""
The Soft Actor-Critic (SAC) algorithm.

* Do not support the vectorized environment.

* Both discrete and continuous action spaces are supported.

references:
- cleanrl: https://docs.cleanrl.dev/rl-algorithms/sac/
- original papers:
    * https://arxiv.org/abs/1801.01290
    * https://arxiv.org/abs/1812.05905

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


class SAC:
    """
    The Soft Actor-Critic (SAC) algorithm.
    """

    def __init__(self, env_id, actor_class, critic_class, exp_name="sac", render=False, seed=1, cuda=0, gamma=0.99,
                 buffer_size=1000000, rb_optimize_memory=False, batch_size=256, policy_lr=3e-4, q_lr=1e-3,
                 alpha_lr=1e-4, target_network_frequency=1, tau=0.005, policy_frequency=2, noise_clip=0.5, alpha=0.2,
                 alpha_autotune=True, write_frequency=100, save_folder="./sac/"):
        """
        Initialize the SAC algorithm.
        :param env_id: the name of the environment
        :param actor_class: the actor class
        :param critic_class: the critic class
        :param exp_name: the name of the experiment
        :param render: whether to render the environment
        :param seed: the random seed
        :param cuda: the cuda device
        :param gamma: the discount factor
        :param buffer_size: the size of the replay buffer
        :param rb_optimize_memory: whether to optimize the memory usage of the replay buffer
        :param batch_size: the batch size
        :param policy_lr: the learning rate of the policy network
        :param q_lr: the learning rate of the Q network
        :param alpha_lr: the learning rate of the temperature parameter
        :param tau: the soft update coefficient
        :param policy_frequency: the policy update frequency
        :param target_network_frequency: the target network update frequency
        :param noise_clip: the noise clip
        :param alpha: the temperature parameter
        :param alpha_autotune: whether to autotune the temperature parameter
        :param write_frequency: the write frequency
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

        self.env = self.make_env(env_id, seed, render)

        # initialize the actor and critic networks
        self.actor = actor_class(self.env).to(self.device)
        self.qf_1 = critic_class(self.env).to(self.device)
        self.qf_2 = critic_class(self.env).to(self.device)
        self.qf_1_target = critic_class(self.env).to(self.device)
        self.qf_2_target = critic_class(self.env).to(self.device)

        # copy the parameters of the critic networks to the target networks
        self.qf_1_target.load_state_dict(self.qf_1.state_dict())
        self.qf_2_target.load_state_dict(self.qf_2.state_dict())

        # initialize the optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=policy_lr)
        self.q_optimizer = optim.Adam(list(self.qf_1.parameters()) + list(self.qf_2.parameters()), lr=q_lr)

        # initialize the temperature parameter
        self.alpha_autotune = alpha_autotune
        if alpha_autotune:
            # set the target entropy as the negative of the action space dimension
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = alpha

        # initialize the replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            optimize_memory_usage=rb_optimize_memory,
            handle_timeout_termination=False
        )

        self.gamma = gamma
        self.batch_size = batch_size

        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.tau = tau
        self.noise_clip = noise_clip

        # * for the tensorboard writer
        run_name = "{}-{}-{}".format(exp_name, seed,
                                     datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S'))
        os.makedirs("./runs/", exist_ok=True)
        self.writer = SummaryWriter(os.path.join("./runs/", run_name))
        self.write_frequency = write_frequency

        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)

    def make_env(self, env_id, seed, render):
        """
        Make the environment.
        :param env_id: the name of the environment
        :param seed: the random seed
        :param render: whether to render the environment
        :return: the environment
        """
        env = gym.make(env_id) if not render else gym.make(env_id, render_mode="human")

        assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported for SAC"

        # TODO: check whether the seed is set
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env

    def learn(self, total_timesteps=1000000, learning_starts=5000):

        obs = self.env.reset()

        for global_step in range(total_timesteps):
            if global_step < learning_starts:
                action = self.env.action_space.sample()
            else:
                action, _, _ = self.actor.get_action(torch.Tensor(obs).to(self.device))
                action = action.detach().cpu().numpy()

            next_obs, reward, terminated, truncated, info = self.env.step(action)

            if "episode" in info:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            self.replay_buffer.add(obs, next_obs, action, reward, terminated, info)

            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > learning_starts:
                self.optimize(global_step)

        self.env.close()
        self.writer.close()

    def optimize(self, global_step):
        data = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(data.next_observations)
            qf_1_next_target = self.qf_1_target(data.next_observations, next_state_actions)
            qf_2_next_target = self.qf_2_target(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf_1_next_target, qf_2_next_target) - self.alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() + \
                           (1 - data.dones.flatten()) * self.gamma * (min_qf_next_target).view(-1)

        qf_1_a_values = self.qf_1(data.observations, data.actions).view(-1)
        qf_2_a_values = self.qf_2(data.observations, data.actions).view(-1)
        qf_1_loss = F.mse_loss(qf_1_a_values, next_q_value)
        qf_2_loss = F.mse_loss(qf_2_a_values, next_q_value)
        qf_loss = qf_1_loss + qf_2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        if global_step % self.policy_frequency == 0:
            for _ in range(self.policy_frequency):
                pi, log_pi, _ = self.actor.get_action(data.observations)
                qf_1_pi = self.qf_1(data.observations, pi)
                qf_2_pi = self.qf_2(data.observations, pi)
                min_qf_pi = torch.min(qf_1_pi, qf_2_pi)
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.alpha_autotune:
                    with torch.no_grad():
                        _, log_pi, _ = self.actor.get_action(data.observations)
                    alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()

        # update the target networks
        if global_step % self.target_network_frequency == 0:
            for param, target_param in zip(self.qf_1.parameters(), self.qf_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf_2.parameters(), self.qf_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if global_step % self.write_frequency == 0:
            self.writer.add_scalar("losses/qf_1_values", qf_1_a_values.mean().item(), global_step)
            self.writer.add_scalar("losses/qf_2_values", qf_2_a_values.mean().item(), global_step)
            self.writer.add_scalar("losses/qf_1_loss", qf_1_loss.item(), global_step)
            self.writer.add_scalar("losses/qf_2_loss", qf_2_loss.item(), global_step)
            self.writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            self.writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            self.writer.add_scalar("losses/alpha", self.alpha, global_step)
            if self.alpha_autotune:
                self.writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    def save(self, indicator="best"):
        if indicator.startswith("best") or indicator.startswith("final"):
            torch.save(self.actor.state_dict(),
                       os.path.join(self.save_folder,
                                    "actor-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed)))
            torch.save(self.qf_1.state_dict(),
                       os.path.join(self.save_folder,
                                    "qf_1-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed)))
            torch.save(self.qf_2.state_dict(),
                       os.path.join(self.save_folder,
                                    "qf_2-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed)))
        else:
            # for normally saved models.
            torch.save(self.actor.state_dict(),
                       os.path.join(self.save_folder,
                                    "actor-{}-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed,
                                                                   datetime.datetime.fromtimestamp(
                                                                       time.time()).strftime(
                                                                       '%Y-%m-%d-%H-%M-%S'))))
            torch.save(self.qf_1.state_dict(),
                       os.path.join(self.save_folder,
                                    "qf_1-{}-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed,
                                                                  datetime.datetime.fromtimestamp(
                                                                      time.time()).strftime(
                                                                      '%Y-%m-%d-%H-%M-%S'))))
            torch.save(self.qf_2.state_dict(),
                       os.path.join(self.save_folder,
                                    "qf_2-{}-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed,
                                                                  datetime.datetime.fromtimestamp(
                                                                      time.time()).strftime(
                                                                      '%Y-%m-%d-%H-%M-%S'))))


class SAC_Atari(SAC):
    """
    The SAC algorithm for Atari games.

    TODO: to complete.
    """

    def __init__(self, env_id, actor_class, critic_class, exp_name="sac", render=False, seed=1, cuda=0, gamma=0.99,
                 buffer_size=1000000, rb_optimize_memory=False, batch_size=256, policy_lr=3e-4, q_lr=3e-4,
                 alpha_lr=1e-4, target_network_frequency=8000, tau=1, policy_frequency=4, noise_clip=0.5, alpha=0.2,
                 alpha_autotune=True, write_frequency=100, save_folder="./sac/"):
        """
        Initialize the SAC algorithm.
        :param env_id: the name of the environment
        :param actor_class: the actor class
        :param critic_class: the critic class
        :param exp_name: the name of the experiment
        :param render: whether to render the environment
        :param seed: the random seed
        :param cuda: the cuda device
        :param gamma: the discount factor
        :param buffer_size: the size of the replay buffer
        :param rb_optimize_memory: whether to optimize the memory usage of the replay buffer
        :param batch_size: the batch size
        :param policy_lr: the learning rate of the policy network
        :param q_lr: the learning rate of the Q network
        :param alpha_lr: the learning rate of the temperature parameter
        :param tau: the soft update coefficient
        :param policy_frequency: the policy update frequency
        :param target_network_frequency: the target network update frequency
        :param noise_clip: the noise clip
        :param alpha: the temperature parameter
        :param alpha_autotune: whether to autotune the temperature parameter
        :param write_frequency: the write frequency
        :param save_folder: the folder to save the model
        """
        super(SAC_Atari, self).__init__(env_id, actor_class, critic_class, exp_name, render, seed, cuda, gamma,
                                        buffer_size, rb_optimize_memory, batch_size, policy_lr, q_lr, alpha_lr,
                                        target_network_frequency, tau, policy_frequency, noise_clip, alpha,
                                        alpha_autotune, write_frequency, save_folder)

    def make_env(self, env_id, seed, render):
        env = gym.make(env_id) if not render else gym.make(env_id, render_mode="human")

        assert isinstance(env.action_space,
                          gym.spaces.Discrete), "only discrete action space is supported for SAC Atari"

        # TODO: check whether the seed is set
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        # some wrappers for the atari environment
        env = gym.wrappers.AtariPreprocessing(env)
        env = gym.wrappers.FrameStack(env, 4)

        # to automatically record the episodic return
        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env
