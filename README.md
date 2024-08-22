# RL Algorithms and Environments by PyTorch

Welcome to the **RL Algo**rithms and **Env**ironments (RLAlgoEnv) repository! This project provides a collection of
reinforcement learning (RL) algorithms implemented in PyTorch and some customized or packed-up environments.

## Table of Contents

- [Requirements](#requirements)
- [Algorithms](#algorithms)
- [Environments](#environments)

## Requirements

- The codes are well tested on `pytorch==2.0.1+cu117`.
- Install all dependent packages:
    ```bash
    pip install -r requirements.txt
    ```
- The project is based the `gymnasium>=0.29.1` package, to render the mujoco, robotics, etc. environments, you need to
  modify the `site-packages\gymnasium\envs\mujoco\mujoco_rendering.py` file: replace the `solver_iter` (at around line
  #593) to `solver_niter`.
- The [running scripts](./run-scripts) can be run directly in [PyCharm](https://www.jetbrains.com/pycharm/), or you may
  need to execute: `export PYTHONPATH=<path to RLEnvsAlgos>:$PYTHONPATH`.

## Algorithms

The project implements RL algorithms in separate independent classes, easy to read and modify.

The implementation of these algorithms is primarily based on the [CleanRL library](https://github.com/vwxyzjn/cleanrl),
which is also an excellent resource that we recommend for reference.

|                   Algorithm                   | Description                                                     | Auther & Year                                                               | Discrete Control | Continuous Control | 
|:---------------------------------------------:|:----------------------------------------------------------------|:----------------------------------------------------------------------------|:----------------:|:------------------:|
|            [DQN](./RLAlgos/DQN.py)            | An enhanced version of Deep Q-Networks algorithm.               | [Mnih et al., 2015](https://www.nature.com/articles/nature14236)            |        ✔️        |         ❌          |
| [CategoricalDQN](./RLAlgos/CategoricalDQN.py) | An extension of DQN with categorical distributional Q-learning. | [Bellemare et al., 2017](https://arxiv.org/abs/1707.06887)                  |        ✔️        |         ❌          |
|      [NoisyNet (DQN)](./RLAlgos/DQN.py)       | An extension of DQN with noisy networks for exploration.        | [Fortunato et al., 2019](https://openreview.net/forum?id=rywHCPkAW)         |        ✔️        |         ❌          |
|            [PPO](./RLAlgos/PPO.py)            | Proximal Policy Optimization.                                   | [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)                   |        ✔️        |       ✔️   ️       |
|            [RPO](./RLAlgos/PPO.py)            | An improved version of PPO.                                     | [Md Masudur Rahman and Yexiang Xue, 2023](https://arxiv.org/abs/2212.07536) |        ✔️        |         ✔️         |
|            [RND](./RLAlgos/RND.py)            | Random Network Distillation, extended from PPO.                 | [Burda et al., 2018](https://openreview.net/forum?id=H1lJJnR5Ym)            |        ✔️        |         ✔️         |
|           [DDPG](./RLAlgos/DDPG.py)           | Deep Deterministic Policy Gradient algorithm.                   | [Silver et al., 2014](https://arxiv.org/abs/1509.02971)                     |        ❌         |         ✔️         |
|            [TD3](./RLAlgos/TD3.py)            | Twin Delayed DDPG, an improved version of DDPG.                 | [Fujimoto et al., 2018](https://proceedings.mlr.press/v80/fujimoto18a)      |        ❌         |         ✔️         |
|            [SAC](./RLAlgos/SAC.py)            | Soft Actor-Critic.                                              | [Haarnoja et al., 2018](https://proceedings.mlr.press/v80/haarnoja18b)      |        ✔️        |         ✔️         |

### Algorithms to be Implemented

- [ ] Soft Q-Learning (SQL)
- [ ] Advantage Actor-Critic (A2C)
- [ ] Asynchronous Advantage Actor-Critic (A3C)
- [ ] and more...

### Running Scripts

We provide a variety of running scripts for different algorithms and environments. You can find
them [here](./run-scripts).

- DQN algorithm (can only work on discrete action spaces):
    * [DQN in classic control](./run-scripts/dqn.py)
    * [DQN in Atari games](./run-scripts/dqn-atari.py)
    * [DQN in MiniGrid environments](./run-scripts/dqn-minigrid.py)
- NoisyNet-DQN algorithm:
    * [NoisyNet-DQN in classic control](./run-scripts/noisynet-dqn.py)
    * [NoisyNet-DQN MiniGrid environments](./run-scripts/noisynet-dqn-minigrid.py)
- PPO algorithm:
    * [PPO in classic control](./run-scripts/ppo.py)
    * [PPO in Atari games](./run-scripts/ppo-atari.py)
    * [PPO in continuous control (e.g., Mujoco)](./run-scripts/ppo-continuous.py)
    * [PPO in robotics control (e.g., gymnasium-robotics, MyFetchRobot)](./run-scripts/ppo-robotics.py)
- RPO algorithm:
    * [RPO in continuous control](./run-scripts/rpo.py)
- RND algorithm:
    * [RND in continuous control](./run-scripts/rnd-continuous.py)
    * [RND in robotics control (e.g., gymnasium-robotics, MyFetchRobot)](./run-scripts/rnd-robotics.py)
    * [RND in MiniGrid environments](./run-scripts/rnd-minigrid.py)
    * [RND in Atari environments](./run-scripts/rnd-atari.py)
- DDPG algorithm (can only work on continuous action spaces):
    * [DDPG in continuous control](./run-scripts/ddpg.py)
- TD3 algorithm (can only work on continuous action spaces):
    * [TD3 in continuous control](./run-scripts/td3.py)
    * [TD3 in robotics control (e.g., gymnasium-robotics, MyFetchRobot)](./run-scripts/td3-robotics.py)
- SAC algorithm:
    * [SAC in continuous control](./run-scripts/sac.py)
    * [SAC in Atari games](./run-scripts/sac-atari.py)
    * [SAC in robotics control (e.g., gymnasium-robotics, MyFetchRobot)](./run-scripts/sac-robotics.py)
    * [SAC in MiniGrid environments](./run-scripts/sac-minigrid.py)

## Environments

We pack up and customize a variety of environments for testing and benchmarking RL algorithms. All environments packages
can be found in [RLEnvs folder](./RLEnvs/).

- [gymnasium](./RLEnvs/gymnasium): the OpenAI [gymnasium](https://gymnasium.farama.org/) library.
- [MyMiniGrid](./RLEnvs/MyMiniGrid): based on the [MiniGrid](https://github.com/Farama-Foundation/MiniGrid) environment,
  customized some wrappers and self-designed environments.
- [MyPandaRobot](./RLEnvs/MyPandaRobot): based on the [panda-gym](https://github.com/qgallouedec/panda-gym/tree/master)
  environment, customized some self-designed environments.
- [MyFetchRobot](./RLEnvs/MyFetchRobot): based on the [gymnasium-robotics](https://robotics.farama.org/index.html)
  library, customized the reward function to only give **sparse and delayed** rewards for four FetchRobot tasks.
- [MyMujoco](./RLEnvs/MyMujoco): based on the [gymnasium-mujoco](https://gymnasium.farama.org/environments/mujoco/)
  library, customized the reward function to only give **sparse and delayed** rewards.
- [MyMiniWorld](./RLEnvs/MyMiniWorld): based on the [MiniWorld](https://miniworld.farama.org/) environment, customized
  some self-designed and sparse-reward environments.

### Templates

We provide some templates to interact with the environments:

- [Using gymnasium](./EnvsTemplates/gymnasium-basic.py)
- [Using gymnasium-robotics](./EnvsTemplates/gymnasium-robotics.py)
- [Using MyMiniGrid](./EnvsTemplates/MyMiniGrid.py)
- [Using MyPandaRobot](./EnvsTemplates/MyPandaRobot.py)
- [Using MyFetchRobot](./EnvsTemplates/MyFetchRobot.py)
- [Using MyMujoco](./EnvsTemplates/MyMujoco.py)

## Contributing

We welcome contributions!

**Actually**, the codes are not thoroughly tested, so we sincerely invite you to help us update the repository. If you
have improvements or bug fixes, please feel free to open an issue or a pull request. Thanks in advance for your help!