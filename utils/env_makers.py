"""
Some functions to initialize environments.
"""

import gymnasium as gym

import numpy as np


def classic_control_env_maker(env_id, seed=1, render=False):
    """
    Make the environment.
    :param env_id: the name of the environment
    :param seed: the random seed
    :param render: whether to render the environment
    :return: the environment
    """
    env = gym.make(env_id) if not render else gym.make(env_id, render_mode="human")

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env


def atari_games_env_maker(env_id, seed=1, render=False):
    env = gym.make(env_id) if not render else gym.make(env_id, render_mode="human")

    assert isinstance(env.action_space, gym.spaces.Discrete), "only discrete action space is supported for Atari Games!"

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # some wrappers for the atari environment
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=1)
    env = gym.wrappers.FrameStack(env, 4)

    # to automatically record the episodic return
    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env


def mujoco_env_maker(env_id, gamma, seed=1, render=False):
    """
    Make the mujoco environment (especially for PPO algorithm).
    :param env_id: the name of the environment
    :param seed: the random seed
    :param render: whether to render the environment
    :return: the environment
    """
    env = gym.make(env_id) if not render else gym.make(env_id, render_mode="human")

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

    return env


def robotics_env_maker(env_id, seed=1, render=False, **kwargs):
    env = gym.make(env_id) if not render else gym.make(env_id, render_mode="human")

    # if kwargs is None:
    #     env = gym.make(env_id) if not render else gym.make(env_id, render_mode="human")
    # else:
    #     env = gym.make(env_id, **kwargs) if not render else gym.make(env_id, render_mode="human", **kwargs)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # + transform the reward from {-1, 0} to {0, 1}
    env = gym.wrappers.TransformReward(env, lambda reward: reward + 1.0)
    # + flatten the dict observation space to a vector
    env = gym.wrappers.TransformObservation(
        env, lambda obs: np.concatenate([obs["observation"], obs["achieved_goal"], obs["desired_goal"]])
    )

    new_obs_length = (
            env.observation_space["observation"].shape[0]
            + env.observation_space["achieved_goal"].shape[0]
            + env.observation_space["desired_goal"].shape[0]
    )

    # redefine the observation of the environment, make it the same size of the flattened dict observation space
    env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(new_obs_length,), dtype=np.float32)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env


def sync_vector_classic_envs_maker(env_id, num_envs, seed=1):
    """
    Make the synchronized vectorized environments.
    :param env_id: the name of the environment
    :param num_envs: the number of environments
    :param seed: the random seed
    :return: the vectorized environments
    """

    envs = gym.vector.SyncVectorEnv(
        [lambda: classic_control_env_maker(env_id, seed, render=False) for _ in range(num_envs)]
    )

    return envs


def sync_vector_atari_envs_maker(env_id, num_envs, seed=1):
    """
    Make the synchronized vectorized environments.
    :param env_id: the name of the environment
    :param num_envs: the number of environments
    :param seed: the random seed
    :return: the vectorized environments
    """

    envs = gym.vector.SyncVectorEnv(
        [lambda: atari_games_env_maker(env_id, seed, render=False) for _ in range(num_envs)]
    )

    return envs


def sync_vector_mujoco_envs_maker(env_id, num_envs, gamma, seed=1):
    """
    Make the synchronized vectorized environments.
    :param env_id: the name of the environment
    :param num_envs: the number of environments
    :param seed: the random seed
    :return: the vectorized environments
    """

    envs = gym.vector.SyncVectorEnv(
        [lambda: mujoco_env_maker(env_id, gamma, seed, render=False) for _ in range(num_envs)]
    )

    return envs


def sync_vector_robotics_envs_maker(env_id, num_envs, seed=1):
    envs = gym.vector.SyncVectorEnv(
        [lambda: robotics_env_maker(env_id, seed, render=True) for _ in range(num_envs)]
    )

    return envs
