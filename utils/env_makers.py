"""
Some functions to initialize environments.
"""

import gymnasium as gym


def classic_control_env_maker(env_id, seed, render):
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


def atari_games_env_maker(env_id, seed, render):
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