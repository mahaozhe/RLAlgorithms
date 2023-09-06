import gymnasium as gym

import RLEnvs.MyPandaRobot.MyPandaRobotEnvs

# + use MiniGrid environments
env_name = "PandaReach-v3"
# + or use self-defined environments
# env_name = "My-Panda-Reach-Target-Sparse"

env = gym.make(env_name)  # no render
# env = gym.make(env_name, render_mode="human")  # auto render, no return from env.render()
# env = gym.make(env_name, render_mode="rgb_array")  # return a (x,y,3) np.ndarray by env.render()

observation, info = env.reset(seed=0)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    # rendered = env.render()   # if `render_mode` is "rgb_array", return a rendered image.

    if terminated or truncated:
        observation, info = env.reset()

env.close()
