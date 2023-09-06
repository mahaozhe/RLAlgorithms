import numpy as np

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.core import Task
from panda_gym.envs.core import RobotTaskEnv

from panda_gym.pybullet import PyBullet

from panda_gym.envs.robots.panda import Panda

from panda_gym.utils import distance

from gymnasium.envs.registration import register


class MyReachTargetTask(Task):
    def __init__(
            self,
            sim,
            get_ee_position,
            reward_type="sparse",
            distance_threshold=0.05,
            goal_range=0.3,
    ) -> None:
        super(MyReachTargetTask, self).__init__(sim)
        self.reward_type = reward_type  # "sparse" or "dense"
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])

        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)  # * create a plane
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)  # * create a table
        # * create a target sphere
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.array([0.1, 0.1, 0.1]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        # * in this function, we can return some task-specific observations
        return np.array([])

    def get_achieved_goal(self) -> np.ndarray:
        # * return the current ee position, the achieved position.
        achieved_goal = np.array(self.get_ee_position())
        return achieved_goal

    def reset(self) -> None:
        self.goal = self._sample_goals()
        self.sim.set_base_pose("target", self.goal[:3], np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goals(self) -> np.ndarray:
        # + this function is used to sample a goal, usually randomly.
        return np.array([0.1, 0.1, 0.1], dtype=np.float32)

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal[:3], desired_goal[:3])
        if d < self.distance_threshold:
            return np.array(True)
        else:
            return np.array(False)

    def compute_reward(self, achieved_goal, desired_goal, info) -> np.ndarray:
        if self.reward_type == "sparse":
            # * for the "sparse" rewards, give a reward only when the final goal is completed
            if self.is_success():
                return np.array(1, dtype=np.float32)
            else:
                return np.array(0, dtype=np.float32)

        else:
            # * for the "dense" rewards, give the reward to assess the distance
            d = distance(achieved_goal[:3], desired_goal[:3])
            return -d.astype(np.float32)


class MyReachTargetEnv(RobotTaskEnv):

    def __init__(
            self,
            render_mode: str = "rgb_array",
            reward_type: str = "sparse",
            control_type: str = "ee",
            renderer: str = "Tiny",
            render_width: int = 720,
            render_height: int = 480,
            render_target_position=None,
            render_distance: float = 1.4,
            render_yaw: float = 45,
            render_pitch: float = -30,
            render_roll: float = 0,
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = MyReachTargetTask(sim, reward_type=reward_type, get_ee_position=robot.get_ee_position)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )


register(
    id="My-Panda-Reach-Target-Sparse",
    entry_point="Envs.PandaRobot.Envs:MyReachTargetEnv",
    kwargs={"reward_type": "sparse"},
    max_episode_steps=200,
)

register(
    id="My-Panda-Reach-Target-Dense",
    entry_point="Envs.PandaRobot.Envs:MyReachTargetEnv",
    kwargs={"reward_type": "dense"},
    max_episode_steps=200,
)
