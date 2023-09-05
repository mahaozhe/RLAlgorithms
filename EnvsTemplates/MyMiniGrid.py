import gymnasium as gym

import RLEnvs.MyMiniGrid.MyMiniGridEnvs

from minigrid.wrappers import *

from RLEnvs.MyMiniGrid.Wrappers import AgentLocation, MovetoFourDirectionsWrapper

# + use MiniGrid environments
env_name = "MiniGrid-Empty-8x8-v0"
# + or use self-defined environments
# env_name = "MiniGrid-Env-Four-Room"

env = gym.make(env_name)  # no render
# env = gym.make(env_name, render_mode="human")  # auto render, no return from env.render()
# env = gym.make(env_name, render_mode="rgb_array")  # return a (x,y,3) np.ndarray by env.render()

# + wrappers
# env = FullyObsWrapper(env)  # Fully observable gridworld instead of the agent view
# env = ImgObsWrapper(env)  # Get rid of the 'mission' field
# env = RGBImgObsWrapper(env, tile_size=8)  # use fully observable RGB image as observation
# env = RGBImgPartialObsWrapper(env, tile_size=8)  # use partially observable RGB image as observation
# env = SymbolicObsWrapper(env)  # fully observable grid with symbolic state representations (not RGB image)
# env = ViewSizeWrapper(env,agent_view_size=7)    # set the view size of the agent

# + self-defined wrappers
# env = AgentLocation(env)  # add the agent location to the `info` with the key `agent_loc`
# env = MovetoFourDirectionsWrapper(env)  # change the action space to make the agent move to four directions directly

observation, info = env.reset(seed=0)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    # rendered = env.render()   # if `render_mode` is "rgb_array", return a rendered image.

    if terminated or truncated:
        observation, info = env.reset()

env.close()
