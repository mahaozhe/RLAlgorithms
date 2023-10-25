"""
The Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm.

* Do not support the vectorized environment.

* Only continuous action spaces are supported.

references:
- cleanrl: https://docs.cleanrl.dev/rl-algorithms/td3/
- cleanrl codes: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py
- original papers:
    * https://arxiv.org/abs/1802.09477

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


class TD3:
    """
    The Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm.
    """

    def __init__(self, env, actor_class, critic_class, exp_name="td3", seed=1, cuda=0, gamma=0.99, buffer_size=1000000,
                 rb_optimize_memory=False, exploration_noise=0.1, policy_regular_noise=0.2, noise_clip=0.5,
                 actor_lr=3e-4, critic_lr=3e-4, batch_size=256, policy_frequency=2, tau=0.005, write_frequency=100,
                 save_folder="./td3/"):
        """
        Initialize the TD3 algorithm.
        :param env: the gymnasium-based environment
        :param actor_class: the actor class
        :param critic_class: the critic class
        :param exp_name: the name of the experiment
        :param seed: the random seed
        :param cuda: the cuda device
        :param gamma: the discount factor
        :param buffer_size: the size of the replay buffer
        :param rb_optimize_memory: whether to optimize the memory usage of the replay buffer
        :param exploration_noise: the exploration noise
        :param policy_regular_noise: the policy smoothing regularization noise scale.
        :param noise_clip: the noise clip
        :param actor_lr: the learning rate of the actor
        :param critic_lr: the learning rate of the critic
        :param batch_size: the batch size
        :param policy_frequency: the policy update frequency
        :param tau: the soft update coefficient
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

        self.env = env

        # initialize the actor and critic networks
        self.actor = actor_class(self.env).to(self.device)
        self.actor_target = actor_class(self.env).to(self.device)

        # twin critic
        self.qf_1 = critic_class(self.env).to(self.device)
        self.qf_2 = critic_class(self.env).to(self.device)
        self.qf_1_target = critic_class(self.env).to(self.device)
        self.qf_2_target = critic_class(self.env).to(self.device)

        # copy the parameters of the policy networks to the target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.qf_1_target.load_state_dict(self.qf_1.state_dict())
        self.qf_2_target.load_state_dict(self.qf_2.state_dict())

        # initialize the optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q_optimizer = optim.Adam(list(self.qf_1.parameters()) + list(self.qf_2.parameters()), lr=critic_lr)

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

        self.exploration_noise = exploration_noise
        self.policy_regular_noise = policy_regular_noise
        self.noise_clip = noise_clip

        self.policy_frequency = policy_frequency
        self.tau = tau

        # * for the tensorboard writer
        run_name = "{}-{}-{}-{}".format(exp_name, env.unwrapped.spec.id, seed,
                                        datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S'))
        os.makedirs("./runs/", exist_ok=True)
        self.writer = SummaryWriter(os.path.join("./runs/", run_name))
        self.write_frequency = write_frequency

        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)

    def learn(self, total_timesteps=1000000, learning_starts=25000):

        obs, _ = self.env.reset()

        for global_step in range(total_timesteps):

            if global_step < learning_starts:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    action = self.actor(torch.Tensor(obs).to(self.device))
                    action += torch.normal(0, self.actor.action_scale * self.exploration_noise)
                    action = action.cpu().numpy().clip(self.env.action_space.low, self.env.action_space.high)

            next_obs, reward, terminated, truncated, info = self.env.step(action)

            if "episode" in info:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            self.replay_buffer.add(obs, next_obs, action, reward, terminated, info)

            if not terminated:
                obs = next_obs
            else:
                obs, _ = self.env.reset()

            if global_step > learning_starts:
                self.optimize(global_step)

        self.env.close()
        self.writer.close()

    def optimize(self, global_step):

        data = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            clipped_noise = (torch.rand_like(data.actions, device=self.device) * self.policy_regular_noise).clamp(
                -self.noise_clip, self.noise_clip) * self.actor.action_scale

            next_state_actions = (self.actor_target(data.next_observations) + clipped_noise).clamp(
                self.env.action_space.low.item(), self.env.action_space.high.item())

            qf1_next_target = self.qf_1_target(data.next_observations, next_state_actions)
            qf2_next_target = self.qf_2_target(data.next_observations, next_state_actions)
            min_q_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * min_q_next_target.view(-1)

        qf1_a_values = self.qf_1(data.observations, data.actions).view(-1)
        qf2_a_values = self.qf_2(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # optimize the model
        self.q_optimizer.zero_grad()
        qf1_loss.backward()
        self.q_optimizer.step()

        if global_step % self.policy_frequency == 0:
            actor_loss = -self.qf_1(data.observations, self.actor(data.observations)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update the target network
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf_1.parameters(), self.qf_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf_2.parameters(), self.qf_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if global_step % self.write_frequency == 0:
            self.writer.add_scalar("losses/qf_1_values", qf1_a_values.mean().item(), global_step)
            self.writer.add_scalar("losses/qf_2_values", qf2_a_values.mean().item(), global_step)
            self.writer.add_scalar("losses/qf_1_loss", qf1_loss.item(), global_step)
            self.writer.add_scalar("losses/qf_2_loss", qf2_loss.item(), global_step)
            self.writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            self.writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)

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
