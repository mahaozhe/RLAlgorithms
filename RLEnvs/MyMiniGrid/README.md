# MiniGrid

- The MiniGrid repo: <https://github.com/Farama-Foundation/Minigrid>
- The MiniGrid document: <https://minigrid.farama.org/>

## Environment Overview

### Observations

The observation returned is a `dict` by default, which commonly contains the following keys:
- `image`: the main observation, there are three kinds:
    * RGB image: `(tile_size*width, tile_size*height, 3)`
    * A 3-dimensional ndarray `(width, height, 3)` where the last dimension contains `(OBJECT_IDX, COLOR_IDX, STATE)`
        ```python
        OBJECT_TO_IDX = {"unseen": 0, "empty": 1, "wall": 2, "floor": 3, "door": 4, "key": 5, "ball": 6, "box": 7, "goal": 8, "lava": 9, "agent": 10}
        COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
        STATE_TO_IDX = {"open": 0, "closed": 1, "locked": 2}
        ```
    * Symbolic state representation: a triple of `(X, Y, IDX)`, where `X` and `Y` are the coordinates on the grid, and `IDX` is the id of the object.
- `direction`: the direction of the agent, `Discrete(4)`.
- `mission`: the string that indicates the current mission/goal for the agent to complete.

### Actions

Original actions:

| Num | Name         | Action                    |
|-----|--------------|---------------------------|
| 0   | left         | Turn left                 |
| 1   | right        | Turn right                |
| 2   | forward      | Move forward              |
| 3   | pickup       | Pick up an object         |
| 4   | drop         | Drop an object            |
| 5   | toggle       | Toggle/activate an object |
| 6   | done         | Done completing task      |

## Wrappers

### MiniGrid Wrappers

```python
from minigrid.wrappers import *
```

- `FullyObsWrapper(env)`: Fully observable gridworld instead of the agent view
- `RGBImgObsWrapper(env, tile_size=8)`: use fully observable RGB image as observation
- `ImgObsWrapper(env)`: Get rid of the 'mission' field
- `RGBImgPartialObsWrapper(env, tile_size=8)`: use partially observable RGB image as observation
- `SymbolicObsWrapper(env)`: fully observable grid with a symbolic state representation
- `ViewSizeWrapper(env,agent_view_size=7)`: set the view size of the agent

The full reference of wrappers: <https://minigrid.farama.org/api/wrappers/>

### Self-Defined Wrappers

```python
from RLEnvs.MyMiniGrid.Wrappers import *
```

- `AgentLocation`: The wrapper to indicate the location of the agent in the `info`.
- `MovetoFourDirectionsWrapper`: Modify the action space to `Discrete(4)`, making the agent only moves to four directions for one step:
    * 0 - move forward
    * 1 - move to left
    * 2 - move to right
    * 3 - move backward

## Reference

```
@article{MinigridMiniworld23,
  author       = {Maxime Chevalier-Boisvert and Bolun Dai and Mark Towers and Rodrigo de Lazcano and Lucas Willems and Salem Lahlou and Suman Pal and Pablo Samuel Castro and Jordan Terry},
  title        = {Minigrid \& Miniworld: Modular \& Customizable Reinforcement Learning Environments for Goal-Oriented Tasks},
  journal      = {CoRR},
  volume       = {abs/2306.13831},
  year         = {2023},
}
```
