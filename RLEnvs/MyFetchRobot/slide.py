import os

from gymnasium.utils.ezpickle import EzPickle
from gymnasium.envs.registration import register

# from .FetchEnv import MujocoFetchEnv
from RLEnvs.MyFetchRobot.FetchEnv import MujocoFetchEnv

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "slide.xml")


class MujocoFetchReachEnv(MujocoFetchEnv, EzPickle):
    """
    ## Description

    """

    def __init__(self, reward_type: str = "sparse", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        MujocoFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=False,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)


register(
    id="MyFetchRobot/Slide-Jnt-Sparse-v0",
    entry_point="RLEnvs.MyFetchRobot.reach:MujocoFetchSlideEnv",
    kwargs={"reward_type": "sparse"},
    max_episode_steps=200,
)
