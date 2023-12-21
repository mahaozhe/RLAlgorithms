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
    * (RPO): https://arxiv.org/abs/2212.07536

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


class PPO:
    """
    The Proximal Policy Optimization (PPO) algorithm.
    """

    def __init__(self, envs, agent_class, exp_name="ppo", seed=1, cuda=0, gamma=0.99, gae_lambda=0.95,
                 rollout_length=128, lr=2.5e-4, eps=1e-5, anneal_lr=True, num_mini_batches=4, update_epochs=4,
                 norm_adv=True, clip_value_loss=True, clip_coef=0.2, entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=0.5, target_kl=None, rpo_alpha=None, write_frequency=100, save_folder="./ppo/"):
        """
        The initialization of the PPO class.
        :param envs: the VECTOR of gymnasium-based environment.
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
        :param num_mini_batches: the number of mini-batches.
        :param update_epochs: the number of update epochs.
        :param norm_adv: whether to normalize the advantages.
        :param clip_value_loss: whether to clip the value loss.
        :param clip_coef: the clipping coefficient.
        :param entropy_coef: the entropy coefficient.
        :param value_coef: the value coefficient.
        :param max_grad_norm: the maximum gradient norm.
        :param target_kl: the target kl divergence.
        :param rpo_alpha: the alpha parameter in RPO.
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

        assert isinstance(envs, gym.vector.SyncVectorEnv), "only vectorized environments are supported!"

        self.envs = envs
        self.num_envs = self.envs.num_envs

        if rpo_alpha is None:
            # using normal PPO agent
            self.agent = agent_class(self.envs).to(self.device)
        else:
            # using RPO agent
            self.agent = agent_class(self.envs, rpo_alpha, cuda=cuda).to(self.device)

        self.anneal_lr = anneal_lr
        self.lr = lr
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr, eps=eps)

        self.rollout_length = rollout_length

        # the big_batch_size is the total timesteps collected in one update: rollout_length * num_envs
        self.big_batch_size = self.rollout_length * self.num_envs
        self.num_mini_batches = num_mini_batches
        # the mini_batch_size is the number of timesteps in one mini-batch: rollout_length * num_envs / num_mini_batches
        self.mini_batch_size = self.big_batch_size // self.num_mini_batches

        # * set up the storage
        self.obs = torch.zeros((self.rollout_length, self.num_envs) + envs.single_observation_space.shape).to(
            self.device)
        self.actions = torch.zeros((self.rollout_length, self.num_envs) + envs.single_action_space.shape).to(
            self.device)
        self.log_probs = torch.zeros((self.rollout_length, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.rollout_length, self.num_envs)).to(self.device)
        self.done = torch.zeros((self.rollout_length, self.num_envs)).to(self.device)
        self.values = torch.zeros((self.rollout_length, self.num_envs)).to(self.device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.update_epochs = update_epochs

        self.norm_adv = norm_adv

        self.clip_value_loss = clip_value_loss
        self.clip_coef = clip_coef
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        # * for the tensorboard writer
        run_name = "{}-{}-{}-{}".format(exp_name, envs.envs[0].unwrapped.spec.id, seed,
                                        datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S'))
        os.makedirs("./runs/", exist_ok=True)
        self.writer = SummaryWriter(os.path.join("./runs/", run_name))
        self.write_frequency = write_frequency

        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)

    def learn(self, total_timesteps=500000):
        global_step = 0

        next_obs, _ = self.envs.reset()
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)

        # the number of updates = total_timesteps // (rollout_length * num_envs)
        num_updates = total_timesteps // self.big_batch_size

        for update in range(1, num_updates + 1):
            # annealing the lr if needed
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lr_now = self.lr * frac
                self.optimizer.param_groups[0]["lr"] = lr_now

            # + iteration the rollout_length steps, store the collected data
            for step in range(self.rollout_length):

                # for each step in the rollout, the global step increases by the number of environments
                global_step += self.num_envs

                # * collect the data
                self.obs[step] = next_obs
                self.done[step] = next_done

                with torch.no_grad():
                    action, log_prob, _, value = self.agent.get_action_value(next_obs)
                    self.values[step] = value.flatten()

                self.actions[step] = action
                self.log_probs[step] = log_prob

                next_obs, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())

                done = np.logical_or(terminated, truncated)
                self.rewards[step] = torch.Tensor(reward).to(self.device).view(-1)

                next_obs = torch.Tensor(next_obs).to(self.device)
                next_done = torch.Tensor(done).to(self.device)

                # check if there is 1 in the next_done
                if next_done.sum() > 0:
                    one_done_index = torch.where(next_done == 1)[0][0]
                    episodic_return = info["final_info"][one_done_index]["episode"]["r"]
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    self.writer.add_scalar("charts/episodic_return", episodic_return, global_step)

            self.optimize(global_step, next_obs, next_done)

        self.envs.close()
        self.writer.close()

    def optimize(self, global_step, next_obs, next_done):
        # bootstrap value
        with torch.no_grad():
            # compute the next value for the last step
            next_value = self.agent.get_value(next_obs).reshape(-1, 1)
            advantages = torch.zeros((self.rollout_length, self.num_envs)).to(self.device)

            last_gae_lam = 0

            for t in reversed(range(self.rollout_length)):
                # if it is the last step, then the next non-terminal value is the bootstrap value
                if t == self.rollout_length - 1:
                    next_non_terminal = 1.0 - next_done
                    next_values = next_value
                # if it is not the last step, then the next non-terminal value is the value of the next step
                else:
                    next_non_terminal = 1.0 - self.done[t + 1]
                    next_values = self.values[t + 1]

                # compute the TD residual: the advantage
                delta = self.rewards[t] + self.gamma * next_values.view(-1) * next_non_terminal - self.values[t]

                advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam

            returns = advantages + self.values

        # flatten the big batch
        b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
        b_log_probs = self.log_probs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # optimize the policy and value networks
        b_indices = np.arange(self.big_batch_size)
        clip_fracs = []

        # run multiple epochs to optimize the policy network
        for epoch in range(self.update_epochs):
            # shuffle the indices of the big batch
            np.random.shuffle(b_indices)
            for start in range(0, self.big_batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = b_indices[start:end]

                _, new_log_probs, entropy, new_values = self.agent.get_action_value(b_obs[mb_indices],
                                                                                    b_actions[mb_indices])
                log_ratio = new_log_probs - b_log_probs[mb_indices]
                ratio = log_ratio.exp()

                # calculate the approximated kl divergence
                with torch.no_grad():
                    old_approx_kl = -log_ratio.mean()
                    approx_kl = (ratio - 1 - log_ratio).mean()
                    clip_fracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_indices]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # calculate the policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # calculate the value loss
                new_values = new_values.view(-1)
                # clip the value loss if needed
                if self.clip_value_loss:
                    v_loss_unclipped = (new_values - b_returns[mb_indices]) ** 2
                    v_clipped = b_values[mb_indices] + torch.clamp(new_values - b_values[mb_indices], -self.clip_coef,
                                                                   self.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_indices]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_values - b_returns[mb_indices]) ** 2).mean()

                # calculate the entropy loss
                entropy_loss = entropy.mean()

                # compute the final loss
                loss = pg_loss - self.entropy_coef * entropy_loss + self.value_coef * v_loss

                # optimize the network
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

            # check the kl divergence
            if self.target_kl is not None and approx_kl > self.target_kl:
                print("Early stopping at step {} due to reaching max kl.".format(epoch))
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # write the logs
        self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
        self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clip_fracs), global_step)
        self.writer.add_scalar("losses/explained_variance", explained_var, global_step)

    def save(self, indicator="best"):
        if indicator.startswith("best") or indicator.startswith("final"):
            torch.save(self.agent.state_dict(),
                       os.path.join(self.save_folder,
                                    "actor-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed)))

        else:
            # for normally saved models.
            torch.save(self.agent.state_dict(),
                       os.path.join(self.save_folder,
                                    "actor-{}-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed,
                                                                   datetime.datetime.fromtimestamp(
                                                                       time.time()).strftime(
                                                                       '%Y-%m-%d-%H-%M-%S'))))
