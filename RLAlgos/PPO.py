"""
The Proximal Policy Optimization (PPO) algorithm.

* support continuous action space
* a `gymnasium` version envs
* support `VectorEnv`

main reference:
- original paper: https://arxiv.org/abs/1707.06347
- CleanRL doc: https://docs.cleanrl.dev/rl-algorithms/ppo/
- CleanRL codes (discrete actions): https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
- CleanRL codes (continuous actions): https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
"""

import os
import random
import time

from tqdm import trange

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


class PPO:
    def __init__(self, env_id, agent_class, render=False, seed=1, cuda=0, learning_rate=3e-4, anneal_lr=True,
                 num_envs=8, parallel=False, num_steps=2048, num_mini_batches=32, gamma=0.99, gae_lambda=0.95,
                 update_epochs=10, clip_coef=0.2, clip_vloss=True, norm_adv=True, entropy_coef=0.0, vf_coef=0.5,
                 max_grad_norm=0.5, target_kl=None, write_frequency=100, save_folder="./runs/"):

        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.envs = self.make_env(env_id, render, num_envs, parallel)

        self.device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

        self.agent = agent_class(self.envs).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate, eps=1e-5)

        self.obs = torch.zeros((num_steps, num_envs) + self.envs.single_observation_space.shape).to(self.device)
        self.actions = torch.zeros((num_steps, num_envs) + self.envs.single_action_space.shape).to(self.device)
        self.log_probs = torch.zeros((num_steps, num_envs)).to(self.device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(self.device)
        self.dones = torch.zeros((num_steps, num_envs)).to(self.device)
        self.values = torch.zeros((num_steps, num_envs)).to(self.device)

        self.num_envs = num_envs
        self.num_steps = num_steps
        self.batch_size = num_steps * num_envs
        self.mini_batch_size = self.batch_size // num_mini_batches

        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr

        # * for actor update:
        self.update_epochs = update_epochs
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.norm_adv = norm_adv
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        # * for the tensorboard writer
        run_name = "{}-{}-{}-{}".format(self.__class__.__name__, env_id, seed, int(time.time()))
        os.makedirs(save_folder, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(save_folder, run_name))
        self.write_frequency = write_frequency

    def make_env(self, env_id, render, num_envs, parallel):
        def make_one():
            env = gym.make(env_id, render_mode="human") if render else gym.make(env_id)

            # * deal with dm_control's Dict observation space
            # envs = gym.wrappers.FlattenObservation(envs)

            # * using the wrapper to record the episode information
            env = gym.wrappers.RecordEpisodeStatistics(env)

            # + the wrappers to normalize the action, observation and rewards
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
            env = gym.wrappers.NormalizeReward(env, gamma=self.gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

            return env

        if parallel:
            # TODO: to make an async envs.
            envs = None
        else:
            envs = gym.vector.SyncVectorEnv([lambda: make_one() for _ in range(num_envs)])

        return envs

    def learn(self, total_timesteps=1000000):
        global_step = 0

        next_obs, _ = self.envs.reset(seed=self.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        num_updates = total_timesteps // self.num_steps

        for update in range(1, num_updates + 1):
            # * annealing the rate
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lr_now = frac * self.learning_rate
                self.optimizer.param_groups[0]['lr'] = lr_now

            for step in range(self.num_steps):
                global_step += self.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done

                # * action selecting logic
                with torch.no_grad():
                    action, log_prob, _, value = self.agent.get_action_and_value(next_obs)
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.log_probs[step] = log_prob

                # * step the environments by the actions
                next_obs, reward, terminated, truncated, infos = self.envs.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs = torch.Tensor(next_obs).to(self.device)
                next_done = torch.Tensor(done).to(self.device)

                # * check if any environment is done
                if "final_info" in infos:
                    for info in infos['final_info']:
                        if info is not None:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # * bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards).to(self.device)
                last_gae = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        next_non_terminal = 1.0 - next_done
                        next_values = next_value
                    else:
                        next_non_terminal = 1.0 - self.dones[t + 1]
                        next_values = self.values[t + 1]
                    delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
                    advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
                returns = advantages + self.values

            # * flatten the batch: note that `advantages` and `returns` have no `self.`
            b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_log_probs = self.log_probs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # * optimize the policy and value network
            b_indices = np.arange(self.batch_size)
            clip_fractions = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_indices)
                for start in range(0, self.batch_size, self.mini_batch_size):
                    end = start + self.mini_batch_size
                    mb_indices = b_indices[start:end]

                    _, new_log_prob, entropy, new_value = self.agent.get_action_and_value(b_obs[mb_indices],
                                                                                          b_actions[mb_indices])
                    log_ratio = new_log_prob - b_log_probs[mb_indices]
                    ratio = log_ratio.exp()

                    with torch.no_grad():
                        # calculate approximated kl, ref: http://joschu.net/blog/kl-approx.html
                        old_approx_kl = -log_ratio.mean()
                        approx_kl = ((ratio - 1) - log_ratio).mean()
                        clip_fractions += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_indices]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # * policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # * value loss
                    new_value = new_value.view(-1)
                    if self.clip_vloss:
                        v_loss_un_clipped = (new_value - b_returns[mb_indices]) ** 2
                        v_clipped = b_values[mb_indices] + torch.clamp(new_value - b_values[mb_indices],
                                                                       -self.clip_coef, self.clip_coef)
                        v_loss_clipped = (v_clipped - b_returns[mb_indices]) ** 2
                        v_loss_max = torch.max(v_loss_un_clipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((new_value - b_returns[mb_indices]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.entropy_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            self.writer.add_scalar("losses/new_approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clip_fraction", np.mean(clip_fractions), global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, global_step)

        self.envs.close()
        self.writer.close()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                                    nn.Tanh(), layer_init(nn.Linear(64, 64)), nn.Tanh(),
                                    layer_init(nn.Linear(64, 1), std=1.0))
        self.actor_mean = nn.Sequential(layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                                        nn.Tanh(), layer_init(nn.Linear(64, 64)), nn.Tanh(),
                                        layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01))
        self.actor_log_std = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    ppo = PPO("MountainCarContinuous-v0", Agent, num_envs=1)
    ppo.learn(100000)
