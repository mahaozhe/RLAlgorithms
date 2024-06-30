import gymnasium as gym
from gymnasium import spaces, utils

from miniworld.entity import COLOR_NAMES, Ball, Box, Key
from miniworld.miniworld import MiniWorldEnv


class PickupObjects2(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Room with multiple objects. The agent collects +1 reward for picking up
    each object. Objects disappear when picked up.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |
    | 3   | move_back                   |
    | 4   | pickup                      |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing a RGB image of what the agents sees.

    ## Rewards:

    +1 when agent picked up object

    ## Arguments

    ```python
    PickupObjects(size=12, num_objs=5)
    ```

    `size`: size of world

    `num_objs`: number of objects

    """

    def __init__(
            self, env_size=12, num_objs=1, obj_size=1.0, en_rand_obj_size=False, obj_pos=(10, 1), agent_pos=(1, 1),
            **kwargs
    ):
        assert env_size >= 2
        self.env_size = env_size
        self.num_objs = num_objs
        if type(obj_size) == dict:
            self.obj_size = obj_size
        elif type(obj_size) == float:
            self.obj_size = {"ball": obj_size, "box": obj_size, "key": obj_size}
        self.en_rand_obj_size = en_rand_obj_size
        if self.en_rand_obj_size:
            print("Random object size enabled")

        self.obj_pos = obj_pos
        self.agent_pos = agent_pos

        MiniWorldEnv.__init__(self, max_episode_steps=400, **kwargs)
        utils.EzPickle.__init__(self, env_size, num_objs, **kwargs)

        # Reduce the action space
        self.action_space = spaces.Discrete(self.actions.pickup + 1)

    def _gen_world(self):
        self.add_rect_room(
            min_x=0,
            max_x=self.env_size,
            min_z=0,
            max_z=self.env_size,
            wall_tex="brick_wall",
            floor_tex="asphalt",
            no_ceiling=True,
        )

        # + only use one Ball object at a specific location
        color = "red"

        self.place_entity(Ball(color=color, size=1.0), pos=[self.obj_pos[0], 0, self.obj_pos[1]])

        self.place_agent(dir=5.5, min_x=self.agent_pos[0], max_x=self.agent_pos[0], min_z=self.agent_pos[1],
                         max_z=self.agent_pos[1])

        self.num_picked_up = 0

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.agent.carrying:
            self.entities.remove(self.agent.carrying)
            self.agent.carrying = None
            self.num_picked_up += 1
            reward = 1

            if self.num_picked_up == self.num_objs:
                termination = True

        return obs, reward, termination, truncation, info


gym.register(
    id="MyMiniWorld/MiniWorld-PickupObjects-Pos-10-1",
    entry_point="RLEnvs.MyMiniWorld.pickupobjects:PickupObjects2",
    max_episode_steps=500,
    kwargs={"env_size": 12, "obj_pos": (10, 1), "agent_pos": (1, 1)},
)

gym.register(
    id="MyMiniWorld/MiniWorld-PickupObjects-Pos-1-10",
    entry_point="RLEnvs.MyMiniWorld.pickupobjects:PickupObjects2",
    max_episode_steps=500,
    kwargs={"env_size": 12, "obj_pos": (1, 10), "agent_pos": (1, 1)},
)

gym.register(
    id="MyMiniWorld/MiniWorld-PickupObjects-Pos-10-10",
    entry_point="RLEnvs.MyMiniWorld.pickupobjects:PickupObjects2",
    max_episode_steps=500,
    kwargs={"env_size": 12, "obj_pos": (10, 10), "agent_pos": (1, 1)},
)

gym.register(
    id="MyMiniWorld/MiniWorld-PickupObjects-Pos-5-10",
    entry_point="RLEnvs.MyMiniWorld.pickupobjects:PickupObjects2",
    max_episode_steps=500,
    kwargs={"env_size": 12, "obj_pos": (5, 10), "agent_pos": (1, 1)},
)

gym.register(
    id="MyMiniWorld/MiniWorld-PickupObjects-Pos-10-5",
    entry_point="RLEnvs.MyMiniWorld.pickupobjects:PickupObjects2",
    max_episode_steps=500,
    kwargs={"env_size": 12, "obj_pos": (10, 5), "agent_pos": (1, 1)},
)