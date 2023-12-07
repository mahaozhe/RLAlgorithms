import os

from gymnasium.utils.ezpickle import EzPickle
from gymnasium.envs.registration import register
import numpy as np
from RLEnvs.MyFetchRobot.FetchEnv import MujocoFetchEnv

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "push.xml")


class MujocoFetchRotateEnv(MujocoFetchEnv, EzPickle):
    """
    ## Description
    """

    def __init__(self, reward_type="sparse", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        MujocoFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.2,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            goal_type="rot",
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)


register(
    id="MyFetchRobot/Rotate-v0",
    entry_point="RLEnvs.MyFetchRobot.rotate:MujocoFetchRotateEnv",
    max_episode_steps=200,
)
