"""
The self-defined MiniGrid environments.
"""

from gymnasium.envs.registration import register

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall, Door, Key
from minigrid.core.mission import MissionSpace


class MyNormalRoomsEnv(MiniGridEnv):
    """
    Self-defined normal room environments.
    """

    def __init__(self, width, height, walls_list, max_steps, agent_init_loc=None, goal_loc=None, **kwargs):
        """
        Init function.
        :param width: the width of the maze.
        :param height: the height of the maze.
        :param walls_list: the list to indicate coordinates of the walls, except the surrounding walls.
        :param max_steps: number of maximal steps for one episode.
        :param agent_init_loc: initial location of the agent, (1, 1) by default.
        :param goal_loc: the location of the goal state.
        """

        self.walls_list = walls_list
        self._agent_default_pos = agent_init_loc if agent_init_loc is not None else (1, 1)
        self._goal_default_pos = goal_loc if goal_loc is not None else (width - 2, height - 2)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super(MyNormalRoomsEnv, self).__init__(mission_space=mission_space, width=width, height=height,
                                               max_steps=max_steps, **kwargs)

    @staticmethod
    def _gen_mission():
        return "reach the goal"

    def _gen_grid(self, width, height):
        # to create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        for wall in self.walls_list:
            self.grid.set(wall[0], wall[1], Wall())

        self.agent_pos = self._agent_default_pos
        self.grid.set(*self._agent_default_pos, None)
        self.agent_dir = 0

        self.put_obj(Goal(), *self._goal_default_pos)


# + For the "four-room" environment:
env_four_room = [(6, 1), (6, 2), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 11), (6, 12),
                 (1, 6), (3, 6), (4, 6), (5, 6), (7, 7), (8, 7), (10, 7), (11, 7)]

register(
    id='MiniGrid-Env-Four-Room',
    entry_point='RLEnvs.MyMiniGrid.MyMiniGridEnvs:MyNormalRoomsEnv',
    kwargs={"width": 13, "height": 13, "walls_list": env_four_room, "max_steps": 1000},
)

# + For the large empty room environment:
register(
    id='MiniGrid-Large-Empty-Room',
    entry_point='RLEnvs.MyMiniGrid.MyMiniGridEnvs:MyNormalRoomsEnv',
    kwargs={"width": 22, "height": 22, "walls_list": [], "max_steps": 1000},
)

# + For the large four-room environment:
env_four_room_large = [(9, 1), (9, 2), (9, 3), (9, 4), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (9, 11), (9, 12),
                       (9, 13), (9, 15), (9, 16), (9, 17),
                       (1, 9), (2, 9), (3, 9), (4, 9), (6, 9), (7, 9), (8, 9), (10, 9), (11, 9), (12, 9), (13, 9),
                       (15, 9), (16, 9), (17, 9)]

register(
    id='MiniGrid-Env-Four-Room-19x19',
    entry_point='RLEnvs.MyMiniGrid.MyMiniGridEnvs:MyNormalRoomsEnv',
    kwargs={"width": 19, "height": 19, "walls_list": env_four_room_large, "max_steps": 1000},
)


# + re-write the Key and Door logics to avoid "pick up" and "toggle"
class MyKey(Key):
    def __init__(self, color="yellow"):
        super().__init__(color)

    def can_overlap(self):
        return True


class MyKeyDoorEnvSelfDefinedKey(MiniGridEnv):
    """
    The class to define an environment with multiple key-doors
    """

    def __init__(self, width, height, objects_loc, max_steps, agent_init_loc=None, goal_loc=None, **kwargs):
        """
        Init function.
        :param width: the width of the maze.
        :param height: the height of the maze.
        :param objects_loc: the dic to indicate where are the walls, doors and keys.
        :param max_steps: number of maximal steps for one episode.
        :param agent_init_loc: initial location of the agent, (1, 1) by default.
        :param goal_loc: the location of the goal state.
        """

        self.walls_list = objects_loc['walls']
        self.door_locs = objects_loc['doors']
        self.key_locs = objects_loc['keys']

        self._agent_default_pos = agent_init_loc if agent_init_loc is not None else (1, 1)
        self._goal_default_pos = goal_loc if goal_loc is not None else (width - 2, height - 2)

        self.colors = ["yellow", "blue", "green"]

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super(MyKeyDoorEnvSelfDefinedKey, self).__init__(mission_space=mission_space, width=width, height=height,
                                                         max_steps=max_steps, **kwargs)

    @staticmethod
    def _gen_mission():
        return "reach the goal"

    def _gen_grid(self, width, height):
        # to create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        self.agent_pos = self._agent_default_pos
        self.grid.set(*self._agent_default_pos, None)
        self.agent_dir = 0

        self.put_obj(Goal(), *self._goal_default_pos)

        # to place the walls
        for wall in self.walls_list:
            self.grid.set(wall[0], wall[1], Wall())

        # to place the door
        for d in range(len(self.door_locs)):
            self.grid.set(self.door_locs[d][0], self.door_locs[d][1], Door(self.colors[d], is_locked=True))

        # to place the key
        for k in range(len(self.key_locs)):
            self.grid.set(self.key_locs[k][0], self.key_locs[k][1], MyKey(self.colors[k]))


# + the "key-door" environment with two pairs of key-doors

env_key_door = {"walls": [(8, 1), (8, 2), (8, 3), (8, 5), (8, 6), (8, 7), (8, 9), (8, 10), (8, 11)],
                "doors": [(8, 4), (8, 8)],
                "keys": [(4, 9), (4, 3)]}

register(
    id='MiniGrid-Env-Two-Key-Door',
    entry_point='RLEnvs.MyMiniGrid.MyMiniGridEnvs:MyKeyDoorEnvSelfDefinedKey',
    kwargs={"width": 13, "height": 13, "objects_loc": env_key_door, "max_steps": 2000, "agent_init_loc": (1, 6),
            "goal_loc": (11, 6)}
)

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
