# MiniGrid

- The MiniGrid repo: <https://github.com/Farama-Foundation/Minigrid>
- The MiniGrid document: <https://minigrid.farama.org/>

## Overview

### Observations

there are mainly three kinds of observations:

* RGB image: `(tile_size*width, tile_size*height, 3)`
* A 3 dimensional tuple `(OBJECT_IDX, COLOR_IDX, STATE)`
    ```python
    COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
    
    OBJECT_TO_IDX = {
        "unseen": 0,
        "empty": 1,
        "wall": 2,
        "floor": 3,
        "door": 4,
        "key": 5,
        "ball": 6,
        "box": 7,
        "goal": 8,
        "lava": 9,
        "agent": 10,
    }
    
    STATE_TO_IDX = {
        "open": 0,
        "closed": 1,
        "locked": 2,
    }
    ```
* Symbolic state representation: a triple of `(X, Y, IDX)`, where `X` and `Y` are the coordinates on the grid, and `IDX` is the id of the object.

### Actions

Original actions:

| Num | Name         | Action       |
|-----|--------------|--------------|
| 0   | left         | Turn left    |
| 1   | right        | Turn right   |
| 2   | forward      | Move forward |
| 3   | pickup       | Unused       |
| 4   | drop         | Unused       |
| 5   | toggle       | Unused       |
| 6   | done         | Unused       |

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