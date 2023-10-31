import gymnasium as gym

from gymnasium.wrappers import *

from utils.env_makers import robotics_env_maker

env_name = "FetchReach-v2"

env = robotics_env_maker(env_id=env_name, render=False)

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
