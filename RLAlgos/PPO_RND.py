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
from stable_baselines3.common.running_mean_std import RunningMeanStd

from Networks.CombinedActorCriticNetworks import RNDModel, PPORNDAgent

import os
import random
import datetime
import time


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RewardForwardFilter:
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


class PPO_RND:
    """
    The Proximal Policy Optimization (PPO) algorithm.
    """

    def __init__(
        self,
        envs,
        agent_class,
        exp_name="ppo",
        seed=1,
        cuda=0,
        gamma=0.99,
        # PND args------
        int_gamma=0.99,
        gae_lambda=0.95,
        int_coef=1.0,
        ext_coef=2.0,
        update_proportion=0.25,
        num_iterations_obs_norm_init=50,
        # --------------
        rollout_length=128,
        lr=2.5e-4,
        eps=1e-5,
        anneal_lr=True,
        num_mini_batches=4,
        update_epochs=4,
        norm_adv=True,
        clip_value_loss=True,
        clip_coef=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        rpo_alpha=None,
        write_frequency=100,
        save_folder="./ppo/",
    ):
        """
        The initialization of the PPO class.
        :param envs: the VECTOR of gymnasium-based environment.
        :param agent_class: the class of the agent.
        :param exp_name: the name of the experiment.
        :param seed: the random seed.
        :param cuda: the cuda device.
        :param gamma: the discount factor.
        :param int_gamma: the discount factor for intrinsic reward.
        :param gae_lambda: the lambda coefficient in generalized advantage estimation.
        :param int_coef: the coefficient for intrinsic reward.
        :param ext_coef: the coefficient for extrinsic reward.
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

        # if rpo_alpha is None:
        #     # using normal PPO agent
        #     self.agent = agent_class(self.envs).to(self.device)
        # else:
        #     # using RPO agent
        #     self.agent = agent_class(self.envs, rpo_alpha, cuda=cuda).to(self.device)

        self.agent = PPORNDAgent(self.envs).to(self.device)

        self.anneal_lr = anneal_lr
        self.lr = lr
        # self.optimizer = optim.Adam(self.agent.parameters(), lr=lr, eps=eps)

        # XXX: RND
        self.int_gamma = int_gamma
        self.int_coef = int_coef
        self.ext_coef = ext_coef
        self.update_proportion = update_proportion
        self.num_iterations_obs_norm_init = num_iterations_obs_norm_init

        self.rnd_model = RNDModel(self.envs).to(self.device)
        self.combined_parameters = list(self.agent.parameters()) + list(self.rnd_model.predictor.parameters())
        self.optimizer = optim.Adam(self.combined_parameters, lr=lr, eps=eps)
        self.reward_rms = RunningMeanStd()
        self.obs_rms = RunningMeanStd(shape=(self.envs.single_observation_space.shape[0]))
        self.discounted_reward = RewardForwardFilter(self.int_gamma)

        self.rollout_length = rollout_length

        # the big_batch_size is the total timesteps collected in one update: rollout_length * num_envs
        self.big_batch_size = self.rollout_length * self.num_envs
        self.num_mini_batches = num_mini_batches
        # the mini_batch_size is the number of timesteps in one mini-batch: rollout_length * num_envs / num_mini_batches
        self.mini_batch_size = self.big_batch_size // self.num_mini_batches

        # * set up the storage
        self.obs = torch.zeros((self.rollout_length, self.num_envs) + envs.single_observation_space.shape).to(
            self.device
        )
        self.actions = torch.zeros((self.rollout_length, self.num_envs) + envs.single_action_space.shape).to(
            self.device
        )
        self.log_probs = torch.zeros((self.rollout_length, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.rollout_length, self.num_envs)).to(self.device)
        self.done = torch.zeros((self.rollout_length, self.num_envs)).to(self.device)
        self.values = torch.zeros((self.rollout_length, self.num_envs)).to(self.device)
        self.curiosity_rewards = torch.zeros((self.rollout_length, self.num_envs)).to(self.device)
        # XXX RND
        self.ext_values = torch.zeros((self.rollout_length, self.num_envs)).to(self.device)
        self.int_values = torch.zeros((self.rollout_length, self.num_envs)).to(self.device)
        # self.avg_returns = deque(maxlen=20)

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
        run_name = "{}-{}-{}-{}".format(
            exp_name,
            envs.envs[0].unwrapped.spec.id,
            seed,
            datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S"),
        )
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

                # action logic
                with torch.no_grad():
                    # action, log_prob, _, value = self.agent.get_action_value(next_obs)
                    # self.values[step] = value.flatten()

                    value_ext, value_int = self.agent.get_value(self.obs[step])
                    self.ext_values[step], self.int_values[step] = (
                        value_ext.flatten(),
                        value_int.flatten(),
                    )
                    action, log_prob, _, _, _ = self.agent.get_action_and_value(self.obs[step])

                self.actions[step] = action
                self.log_probs[step] = log_prob

                next_obs, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())

                done = np.logical_or(terminated, truncated)
                self.rewards[step] = torch.Tensor(reward).to(self.device).view(-1)

                next_obs = torch.Tensor(next_obs).to(self.device)
                next_done = torch.Tensor(done).to(self.device)

                # XXX RND
                # rnd_next_obs = (
                #     (
                #         (
                #             next_obs[:, 3, :, :].reshape(self.num_envs, 1, 84, 84)
                #             - torch.from_numpy(self.obs_rms.mean).to(self.device)
                #         )
                #         / torch.sqrt(torch.from_numpy(self.obs_rms.var).to(self.device))
                #     ).clip(-5, 5)
                # ).float()
                rnd_next_obs = (
                    (
                        (next_obs - torch.from_numpy(self.obs_rms.mean).to(self.device))
                        / torch.sqrt(torch.from_numpy(self.obs_rms.var).to(self.device))
                    )
                    .clip(-5, 5)  # TODO: check whether need to clip
                    .float()
                )
                target_next_feature = self.rnd_model.target(rnd_next_obs)
                predict_next_feature = self.rnd_model.predictor(rnd_next_obs)
                self.curiosity_rewards[step] = ((target_next_feature - predict_next_feature).pow(2).sum(1) / 2).data

                # check if there is 1 in the next_done
                if next_done.sum() > 0:
                    one_done_index = torch.where(next_done == 1)[0][0]
                    episodic_return = info["final_info"][one_done_index]["episode"]["r"]
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    self.writer.add_scalar("charts/episodic_return", episodic_return, global_step)

            # XXX RND
            curiosity_reward_per_env = np.array(
                [
                    self.discounted_reward.update(reward_per_step)
                    for reward_per_step in self.curiosity_rewards.cpu().data.numpy().T
                ]
            )
            mean, std, count = (
                np.mean(curiosity_reward_per_env),
                np.std(curiosity_reward_per_env),
                len(curiosity_reward_per_env),
            )
            self.reward_rms.update_from_moments(mean, std**2, count)
            self.curiosity_rewards /= np.sqrt(self.reward_rms.var)

            self.optimize(global_step, next_obs, next_done)

        self.envs.close()
        self.writer.close()

    def optimize(self, global_step, next_obs, next_done):
        # bootstrap value
        with torch.no_grad():
            # compute the next value for the last step
            # next_value = self.agent.get_value(next_obs).reshape(-1, 1)
            # advantages = torch.zeros((self.rollout_length, self.num_envs)).to(self.device)
            # last_gae_lam = 0

            # XXX RND
            next_value_ext, next_value_int = self.agent.get_value(next_obs)
            next_value_ext, next_value_int = next_value_ext.reshape(1, -1), next_value_int.reshape(1, -1)
            ext_advantages = torch.zeros_like(self.rewards, device=self.device)
            int_advantages = torch.zeros_like(self.curiosity_rewards, device=self.device)
            ext_lastgaelam = 0
            int_lastgaelam = 0

            for t in reversed(range(self.rollout_length)):
                # if it is the last step, then the next non-terminal value is the bootstrap value
                if t == self.rollout_length - 1:
                    # next_non_terminal = 1.0 - next_done
                    # next_values = next_value
                    ext_nextnonterminal = 1.0 - next_done
                    int_nextnonterminal = 1.0
                    ext_nextvalues = next_value_ext
                    int_nextvalues = next_value_int
                # if it is not the last step, then the next non-terminal value is the value of the next step
                else:
                    # next_non_terminal = 1.0 - self.done[t + 1]
                    # next_values = self.values[t + 1]
                    ext_nextnonterminal = 1.0 - self.done[t + 1]
                    int_nextnonterminal = 1.0
                    ext_nextvalues = self.ext_values[t + 1]
                    int_nextvalues = self.int_values[t + 1]

                # compute the TD residual: the advantage
                # delta = self.rewards[t] + self.gamma * next_values.view(-1) * next_non_terminal - self.values[t]

                # advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam

                # XXX RND
                ext_delta = self.rewards[t] + self.gamma * ext_nextvalues * ext_nextnonterminal - self.ext_values[t]
                int_delta = (
                    self.curiosity_rewards[t]
                    + self.int_gamma * int_nextvalues * int_nextnonterminal
                    - self.int_values[t]
                )
                ext_advantages[t] = ext_lastgaelam = (
                    ext_delta + self.gamma * self.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                )
                int_advantages[t] = int_lastgaelam = (
                    int_delta + self.int_gamma * self.gae_lambda * int_nextnonterminal * int_lastgaelam
                )

            # returns = advantages + self.values
            ext_returns = ext_advantages + self.ext_values
            int_returns = int_advantages + self.int_values

        # flatten the big batch
        b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
        b_log_probs = self.log_probs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
        # b_advantages = advantages.reshape(-1)
        # b_returns = returns.reshape(-1)
        # b_values = self.values.reshape(-1)
        b_ext_advantages = ext_advantages.reshape(-1)
        b_int_advantages = int_advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_ext_values = self.ext_values.reshape(-1)
        b_advantages = b_int_advantages * self.int_coef + b_ext_advantages * self.ext_coef

        # self.obs_rms.update(b_obs[:, 3, :, :].reshape(-1, 1, 84, 84).cpu().numpy())
        self.obs_rms.update(b_obs.cpu().numpy())

        # optimize the policy and value networks
        b_indices = np.arange(self.big_batch_size)
        clip_fracs = []

        # rnd_next_obs = (
        #     (
        #         (b_obs[:, 3, :, :].reshape(-1, 1, 84, 84) - torch.from_numpy(self.obs_rms.mean).to(self.device))
        #         / torch.sqrt(torch.from_numpy(self.obs_rms.var).to(self.device))
        #     ).clip(-5, 5)
        # ).float()
        rnd_next_obs = (
            (
                (b_obs - torch.from_numpy(self.obs_rms.mean).to(self.device))
                / torch.sqrt(torch.from_numpy(self.obs_rms.var).to(self.device))
            )
            .clip(-5, 5)
            .float()
        )

        # run multiple epochs to optimize the policy network
        for epoch in range(self.update_epochs):
            # shuffle the indices of the big batch
            np.random.shuffle(b_indices)
            for start in range(0, self.big_batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = b_indices[start:end]

                predict_next_state_feature, target_next_state_feature = self.rnd_model(rnd_next_obs[mb_indices])
                forward_loss = F.mse_loss(
                    predict_next_state_feature, target_next_state_feature.detach(), reduction="none"
                ).mean(-1)
                mask = torch.rand(len(forward_loss), device=self.device)
                mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
                forward_loss = (forward_loss * mask).sum() / torch.max(
                    mask.sum(), torch.tensor([1], device=self.device, dtype=torch.float32)
                )
                _, new_log_probs, entropy, new_ext_values, new_int_values = self.agent.get_action_and_value(
                    b_obs[mb_indices], b_actions[mb_indices]
                )
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
                # new_values = new_values.view(-1)
                new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)
                
                # clip the value loss if needed
                if self.clip_value_loss:
                    # v_loss_unclipped = (new_values - b_returns[mb_indices]) ** 2
                    # v_clipped = b_values[mb_indices] + torch.clamp(
                    #     new_values - b_values[mb_indices], -self.clip_coef, self.clip_coef
                    # )
                    # v_loss_clipped = (v_clipped - b_returns[mb_indices]) ** 2
                    # v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    # v_loss = 0.5 * v_loss_max.mean()
                    
                    ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_indices]) ** 2
                    ext_v_clipped = b_ext_values[mb_indices] + torch.clamp(
                        new_ext_values - b_ext_values[mb_indices],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[mb_indices]) ** 2
                    ext_v_loss_max = torch.max(ext_v_loss_unclipped, ext_v_loss_clipped)
                    ext_v_loss = 0.5 * ext_v_loss_max.mean()
                else:
                    # v_loss = 0.5 * ((new_values - b_returns[mb_indices]) ** 2).mean()
                    ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_indices]) ** 2).mean()

                # calculate the entropy loss
                entropy_loss = entropy.mean()
                int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_indices]) ** 2).mean()
                v_loss = ext_v_loss + int_v_loss
                entropy_loss = entropy.mean()

                # compute the final loss
                # loss = pg_loss - self.entropy_coef * entropy_loss + self.value_coef * v_loss
                loss = pg_loss - self.entropy_coef * entropy_loss + v_loss * self.value_coef + forward_loss

                # optimize the network
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.combined_parameters, self.max_grad_norm)
                self.optimizer.step()

            # check the kl divergence
            if self.target_kl is not None and approx_kl > self.target_kl:
                print("Early stopping at step {} due to reaching max kl.".format(epoch))
                break

        # y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        y_pred, y_true = b_ext_values.cpu().numpy(), b_ext_returns.cpu().numpy()
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
            torch.save(
                self.agent.state_dict(),
                os.path.join(self.save_folder, "actor-{}-{}-{}.pth".format(self.exp_name, indicator, self.seed)),
            )

        else:
            # for normally saved models.
            torch.save(
                self.agent.state_dict(),
                os.path.join(
                    self.save_folder,
                    "actor-{}-{}-{}-{}.pth".format(
                        self.exp_name,
                        indicator,
                        self.seed,
                        datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S"),
                    ),
                ),
            )
