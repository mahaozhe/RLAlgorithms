import gymnasium as gym
from gymnasium import utils

from miniworld.entity import MeshEnt
from miniworld.miniworld import MiniWorldEnv


class CollectHealth2(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Environment where the agent has to collect health kits and stay
    alive as long as possible. This is inspired from the VizDoom
    `HealthGathering` environment. Please note, however, that the rewards
    produced by this environment are not directly comparable to those
    of the VizDoom environment.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |
    | 3   | move back                   |
    | 4   | pick up                     |
    | 5   | drop                        |
    | 6   | toggle / activate an object |
    | 7   | complete task               |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing a RGB image of what the agents sees.

    ## Rewards:

    +2 for each time step
    -100 for dying

    ## Arguments

    ```python
    CollectHealth(size=16)
    ```

    `size`: size of the room

    """

    def __init__(self, env_size=16, num_objs=18, obj_size=0.4, en_rand_obj_size=False, **kwargs):
        assert env_size >= 2
        self.env_size = env_size
        self.num_objs = num_objs
        self.obj_size = obj_size
        self.en_rand_obj_size = en_rand_obj_size

        MiniWorldEnv.__init__(self, max_episode_steps=1000, **kwargs)
        utils.EzPickle.__init__(self, env_size, **kwargs)

    def _gen_world(self):
        # Create a long rectangular room
        self.add_rect_room(
            min_x=0,
            max_x=self.env_size,
            min_z=0,
            max_z=self.env_size,
            wall_tex="cinder_blocks",
            floor_tex="slime",
        )

        for _ in range(self.num_objs):
            self.box = self.place_entity(
                MeshEnt(
                    mesh_name="medkit",
                    height=self.obj_size if not self.en_rand_obj_size else self.np_random.uniform(0.3, 0.5),
                    static=False,
                )
            )

        # Place the agent a random distance away from the goal
        self.place_agent()

        self.health = 100

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        self.health -= 2

        # If the agent picked up a health kit
        if action == self.actions.pickup:
            if self.agent.carrying:
                # Respawn the health kit
                self.entities.remove(self.agent.carrying)
                self.place_entity(self.agent.carrying)
                self.agent.carrying = None

                # Reset the agent's health
                self.health = 100

        if self.health > 0:
            reward = 2
        else:
            reward = -100
            termination = True

        # Pass current health value in info dict
        info["health"] = self.health

        return obs, reward, termination, truncation, info


gym.register(
    id="MyMiniWorld/MiniWorld-CollectHealth",
    entry_point="RLEnvs.MyMiniWorld.collecthealth:CollectHealth2",
    kwargs={"env_size": 16, "num_objs": 10, "obj_size": 0.8},
)