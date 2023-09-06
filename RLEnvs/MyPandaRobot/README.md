# Panda-Gym

- The Panda-Gym repo: <https://github.com/qgallouedec/panda-gym/>
- The MiniGrid document: <https://panda-gym.readthedocs.io/en/latest/index.html>

## Environment Overview

There are two control modes:
- end-effector control: `"ee"` envs: the action is the displacement of the end-effector.
- joint control: `"joint"` envs: the action is the motion of the joints.

### Observations

By default, the observation has three fields:
* `"observation"`: 0-5: robot observation, 6-8: object position, 9-11: object velocity
* `"achieved_goal"`: 0-2: object position
* `"desired_goal"`: 0-2: goal position

### Actions

- For the `"ee"` envs: `Box(-1,1,(3,),float32)`.
- For the `"joint"` envs: `Box(-1,1,(7,),float32)`.

### Rewards

- Sparse: the environment return a reward if and only if the task is completed.
- Dense: the closer the agent is to complete the task, the higher the reward.

## Important

We need to modify the `core.py` script in the `panda-gym` library, at **line 236**:

```python
desired_goal_shape = observation["desired_goal"].shape
```

and the **line 240-241**:

```python
desired_goal=spaces.Box(-10.0, 10.0, shape=desired_goal_shape, dtype=np.float32),
achieved_goal=spaces.Box(-10.0, 10.0, shape=achieved_goal_shape, dtype=np.float32),
```