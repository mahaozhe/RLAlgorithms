from gymnasium.envs.registration import register

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall, Door, Key
from minigrid.core.mission import MissionSpace

AvailableColors = ["yellow", "blue", "green", "red", "purple"]


class MyDoorKeyEnv(MiniGridEnv):
    """
    Environment with a key (keys) and the corresponding door (doors).
    """

    def __init__(self, width, height, obj_lists, max_steps, agent_init_loc=None, goal_loc=None, **kwargs):
        self.obj_lists = obj_lists
        self._agent_default_pos = agent_init_loc if agent_init_loc is not None else (1, 1)
        self._goal_default_pos = goal_loc if goal_loc is not None else (width - 2, height - 2)

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super(MyDoorKeyEnv, self).__init__(mission_space=mission_space, width=width, height=height, max_steps=max_steps,
                                           **kwargs)

    @staticmethod
    def _gen_mission():
        return "use the key to open the door and then get to the goal"

    def _gen_grid(self, width, height):
        # to create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.agent_pos = self._agent_default_pos
        self.grid.set(*self._agent_default_pos, None)
        self.agent_dir = 0

        self.put_obj(Goal(), *self._goal_default_pos)

        for wall in self.obj_lists['walls']:
            self.grid.set(wall[0], wall[1], Wall())

        for d in range(len(self.obj_lists['doors'])):
            self.put_obj(Door(AvailableColors[d], is_locked=True),
                         self.obj_lists['doors'][d][0], self.obj_lists['doors'][d][1])

        for k in range(len(self.obj_lists['keys'])):
            self.place_obj(obj=Key(AvailableColors[k]),
                           top=(self.obj_lists['keys'][k][0], self.obj_lists['keys'][k][1]),
                           size=(1, 1))


empty_room = {"walls": [],
              "doors": [],
              "keys": []}

register(
    id='MiniGrid-Empty-Room',
    entry_point='RLEnvs.MyMiniGrid.MyKeyDoorEnvs:MyDoorKeyEnv',
    kwargs={"width": 12, "height": 12, "obj_lists": empty_room, "max_steps": 500}
)

vertical_wall_1 = {"walls": [(7, 1), (7, 2), (7, 3), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10)],
                   "doors": [(7, 4)],
                   "keys": [(4, 7)]}

register(
    id='MiniGrid-Vertical-Wall-Type1',
    entry_point='RLEnvs.MyMiniGrid.MyKeyDoorEnvs:MyDoorKeyEnv',
    kwargs={"width": 12, "height": 12, "obj_lists": vertical_wall_1, "max_steps": 500}
)

vertical_wall_2 = {"walls": [(6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 9), (6, 10)],
                   "doors": [(6, 8)],
                   "keys": [(3, 5)]}

register(
    id='MiniGrid-Vertical-Wall-Type2',
    entry_point='RLEnvs.MyMiniGrid.MyKeyDoorEnvs:MyDoorKeyEnv',
    kwargs={"width": 12, "height": 12, "obj_lists": vertical_wall_2, "max_steps": 500, "goal_loc": (10, 1)}
)

horizontal_wall_1 = {"walls": [(1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (9, 7), (10, 7)],
                     "doors": [(8, 7)],
                     "keys": [(4, 4)]}

register(
    id='MiniGrid-Horizontal-Wall-Type1',
    entry_point='RLEnvs.MyMiniGrid.MyKeyDoorEnvs:MyDoorKeyEnv',
    kwargs={"width": 12, "height": 12, "obj_lists": horizontal_wall_1, "max_steps": 500}
)

horizontal_wall_2 = {"walls": [(1, 6), (2, 6), (3, 6), (5, 6), (6, 6), (7, 6), (8, 6), (9, 6), (10, 6)],
                     "doors": [(4, 6)],
                     "keys": [(8, 4)]}

register(
    id='MiniGrid-Horizontal-Wall-Type2',
    entry_point='RLEnvs.MyMiniGrid.MyKeyDoorEnvs:MyDoorKeyEnv',
    kwargs={"width": 12, "height": 12, "obj_lists": horizontal_wall_2, "max_steps": 500, "goal_loc": (1, 10)}
)
