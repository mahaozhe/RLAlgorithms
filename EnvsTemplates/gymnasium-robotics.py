import gymnasium as gym

from gymnasium.wrappers import *

env_name = "FetchReach-v2"

env = gym.make(env_name)  # no render
# env = gym.make(env_name, render_mode="human")  # auto render, no return from env.render()
# env = gym.make(env_name, render_mode="rgb_array")  # return a (x,y,3) np.ndarray by env.render()

# + transform the reward from {-1, 0} to {0, 1}
env = gym.wrappers.TransformReward(env, lambda reward: reward + 1.0)

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
