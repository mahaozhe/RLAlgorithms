"""
The script to run SAC on continuous control environments.
"""
import argparse

from Networks.ActorNetworks import SACActor
from Networks.QValueNetworks import QNetworkContinuousControl, QNetworkContinuousControlNew
from RLAlgos.SAC import SAC
from RLEnvs.MyFetchRobot import push, reach, slide, rotate
from RLEnvs.MyMujoco import ant_v4, humanoid_v4, humanoidstandup_v4, reacher_v4, hopper_v4, walker2d_v4
from utils.env_makers import robotics_env_maker

import gymnasium as gym


env_name = "Mujoco/Ant-v4-Sparse"

env = robotics_env_maker(env_id=env_name, render=True, reward_type="sparse", task="pos", goal_dist_th=0.1)

observation, info = env.reset(seed=42)
for i in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print(i)
        observation, info = env.reset()

env.close()

