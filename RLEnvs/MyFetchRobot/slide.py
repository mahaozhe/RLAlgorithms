import os

import numpy as np
from gymnasium.utils.ezpickle import EzPickle
from gymnasium.envs.registration import register

# from .FetchEnv import MujocoFetchEnv
from RLEnvs.MyFetchRobot.FetchEnv import MujocoFetchEnv

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "slide.xml")


class MujocoFetchSlideEnv(MujocoFetchEnv, EzPickle):
    """
    ## Description
    """

    def __init__(self, reward_type="sparse", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.05,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.7, 1.1, 0.41, 1.0, 0.0, 0.0, 0.0],
        }
        MujocoFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=-0.02,
            target_in_the_air=False,
            target_offset=np.array([0.4, 0.0, 0.0]),
            obj_range=0.1,
            target_range=0.3,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)


register(
    id="MyFetchRobot/Slide-Jnt-Sparse-v0",
    entry_point="RLEnvs.MyFetchRobot.slide:MujocoFetchSlideEnv",
    kwargs={"reward_type": "sparse"},
    max_episode_steps=200,
)
