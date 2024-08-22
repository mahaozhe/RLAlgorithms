import gymnasium as gym
from PIL import Image

from RLEnvs.MyMiniWorld import collecthealth, pickupobjects

from RLEnvs.MyMiniWorld.Wrappers import RGBImgObsRevChannelWrapper

env_name = "MyMiniWorld/MiniWorld-CollectHealth"

env = gym.make(env_name)  # no render
# env = gym.make(env_name, render_mode="human")  # auto render, no return from env.render()
# env = gym.make(env_name, render_mode="rgb_array")  # return a (x,y,3) np.ndarray by env.render()

env = RGBImgObsRevChannelWrapper(env)   # reverse the shape of observation from (h,w,c) to (c,h,w)

observation, info = env.reset(seed=0)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    # rendered = env.render()
    # image = Image.fromarray(observation)
    # image.show()

    if terminated or truncated:
        observation, info = env.reset()

env.close()
