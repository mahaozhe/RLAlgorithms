"""
The Soft Actor-Critic (SAC) algorithm.

* support continuous action space
* discrete action space for Atari games
* not support vectorized envs
* a `gymnasium` version env

main reference:
- original paper: https://proceedings.mlr.press/v80/haarnoja18b
- CleanRL doc: https://docs.cleanrl.dev/rl-algorithms/sac/
- CleanRL codes: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
- CleanRL codes (for Atari): https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_atari.py
- OpenAI spinning up: https://spinningup.openai.com/en/latest/algorithms/sac.html
"""

import os
import time
import random
import argparse

import gymnasium as gym
import numpy as np

from tqdm import trange

from stable_baselines3.common.buffers import ReplayBuffer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


# TODO: set seeds

class SAC:
    def __init__(self, env_id, actor_class, qf_class, render=False, seed=1, replay_buffer_size=int(1e6), gamma=0.99,
                 tau=0.005, batch_size=256, policy_lr=3e-4, qf_lr=1e-3, policy_frequency=2, target_network_frequency=1,
                 autotune=True, alpha=0.2, write_frequency=100, save_folder="./runs/"):

        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.env = self.make_env(env_id, render)
        # self.env_name = env_name

        # TODO: to support cuda:N
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.replay_buffer = ReplayBuffer(
            replay_buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            optimize_memory_usage=True,
            handle_timeout_termination=False
        )

        self.actor = actor_class(self.env).to(self.device)
        self.qf1 = qf_class(self.env).to(self.device)
        self.qf2 = qf_class(self.env).to(self.device)

        self.qf1_target = qf_class(self.env).to(self.device)
        self.qf2_target = qf_class(self.env).to(self.device)

        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=policy_lr)
        self.qf_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=qf_lr)

        self.autotune = autotune

        if autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=qf_lr)
        else:
            self.alpha = alpha

        self.batch_size = batch_size

        self.gamma = gamma

        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.tau = tau

        # * for the tensorboard writer
        run_name = "{}-{}-{}".format(env_id, seed, int(time.time()))
        os.makedirs(save_folder, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(save_folder, run_name))
        self.write_frequency = write_frequency

    def make_env(self, env_id, render):
        env = gym.make(env_id, render_mode="human") if render else gym.make(env_id)
        # * using the wrapper to record the episode information
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # + for the Mujoco environments, change the observation space dtype from float64 to float32
        if not np.issubdtype(env.observation_space.dtype, np.float32):
            env.observation_space.dtype = np.float32

        env.action_space.seed(self.seed)
        env.observation_space.seed(self.seed)
        return env

    def learn(self, total_time_steps=int(1e6), learning_starts=5e3):
        # start the game
        obs, info = self.env.reset()

        for global_step in trange(total_time_steps):

            # * action logic:
            if global_step < learning_starts:
                action = self.env.action_space.sample()
            else:
                action, _, _ = self.actor.get_action(torch.Tensor(obs).to(self.device))
                # ! the output of the network has an additional dimension, need the reshape(-1)
                action = action.detach().cpu().numpy().reshape(-1)

            # execute the action
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            if "episode" in info.keys():
                # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # save data to reply buffer
            self.replay_buffer.add(obs, next_obs, action, reward, terminated or truncated, info)

            # * copy the `next_obs` to `obs`
            obs = next_obs

            # train if the learning starts
            if global_step > learning_starts:
                self.optimize(global_step=global_step)

    def optimize(self, global_step):
        # draw a batch of samples from the replay buffer
        data = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(data.next_observations)
            qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
            qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * min_qf_next_target.view(
                -1)

        qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
        qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        # * delayed update
        if global_step % self.policy_frequency == 0:
            # + update multiple times for the delayed update
            for _ in range(self.policy_frequency):
                pi, log_pi, _ = self.actor.get_action(data.observations)
                qf1_pi = self.qf1(data.observations, pi)
                qf2_pi = self.qf2(data.observations, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = self.actor.get_action(data.observations)
                    alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()

                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()

        # * update the target networks
        if global_step % self.target_network_frequency == 0:
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if global_step % self.write_frequency == 0:
            self.writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            self.writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            self.writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            self.writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            self.writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            self.writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            self.writer.add_scalar("losses/alpha", self.alpha, global_step)
            if self.autotune:
                self.writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    def close(self):
        self.env.close()
        self.writer.close()


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer("action_scale",
                             torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias",
                             torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32))

    def forward(self, x):
        # ! need to check whether the input `x` is only one-dimensional
        x = x.unsqueeze(0) if x.dim() == 1 else x

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-id", type=str, default="Ant-v4", help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="total timesteps of the experiments")
    parser.add_argument("--learning-starts", type=int, default=5000, help="timestep to start learning")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    # env_id = "Pendulum-v1"

    sac_agent = SAC(env_id=args.env_id, actor_class=Actor, qf_class=SoftQNetwork)
    sac_agent.learn(total_time_steps=args.total_timesteps, learning_starts=args.learning_starts)
