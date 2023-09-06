# Gymnasium

- [Gymnasium documentation](https://gymnasium.farama.org/)
- [Gym documentation (old version)](https://www.gymlibrary.dev/)
- [Gymnasium source code](https://github.com/Farama-Foundation/Gymnasium)

## Basic Environments

There are different batches of environments:

- Classic Control
- Box2D
- Toy Text
- Atari
- MuJoCo

## Overview

### Spaces

There are different spaces for both observations and actions:

- Discrete
- Box
- MultiDiscrete
- MultiBinary

### Wrappers

There are multiple [available wrappers](https://gymnasium.farama.org/api/wrappers/#gymnasium-wrappers) defined by the gymnasium library, which can be imported from `gymnasium.wrappers`.

Some useful wrappers:

- `RecordEpisodeStatistics`: Records episode statistics (e.g. episode length, episode reward, etc.), at the end of the episode, the statistics will be added in the `info` with the key `episode`.
- `RecordVideo`: This wrapper will record videos of rollouts.
- `AutoResetWrapper`: This wrapper automatically resets the environment when the terminated or truncated state is reached.
- `ClipAction`: This wrapper clips the actions to the given range.
- `NormalizeReward`: This wrapper normalizes the reward to the given range.
- `PixelObservationWrapper`: This wrapper converts the observation to a pixel observation.
- `RescaleAction`: This wrapper rescales the action to the given range.
- `ResizeObservation`: This wrapper works on environments with image observations and resizes the observation to the shape given by the tuple shape.
- `GrayScaleObservation`: Convert the image observation from RGB to gray scale.
- `FrameStack`: This wrapper stacks the observations over a given number of steps.
- `AtariPreprocessing`: Implements the common preprocessing applied tp Atari environments
- `TransformObservation`: This wrapper applies function to observations.
- `TransformReward`: This wrapper applies function to rewards.