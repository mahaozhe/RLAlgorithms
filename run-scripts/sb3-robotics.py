import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import argparse

import sys_env
from Networks.ActorNetworks import SACActor
from Networks.QValueNetworks import QNetworkContinuousControl, QNetworkContinuousControlNew
from RLAlgos.SAC import SAC
from RLEnvs.MyFetchRobot import push, reach, slide, rotate
from RLEnvs.Mujoco import ant_v4, humanoid_v4, humanoidstandup_v4
from utils.env_makers import robotics_env_maker

import gymnasium as gym


class EventCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)

    def _on_step(self) -> bool:
        print(f"step: {self.locals['n_steps']}, rewards: {self.locals['rewards'][0]}")
        return super()._on_step()


env = gym.make("MyFetchRobot/Push-Jnt-v0", render_mode="human", reward_type="dense")
model = PPO(policy="MultiInputPolicy", env=env)
model.learn(30_000, callback=EventCallback())
