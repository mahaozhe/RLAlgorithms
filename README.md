# RL Experimental Environments

## Environments

All RL environments packages can be found in [RLEnvs](./RLEnvs/) package.

- [MyMiniGrid](./RLEnvs/MyMiniGrid)
- [Panda Robot](./RLEnvs/MyPandaRobot)
- [gymnasium](./RLEnvs/gymnasium)
- [MyFetchRobot](./RLEnvs/MyFetchRobot)
- [MyMujoco](./RLEnvs/MyMujoco)

### Templates

- [Using MyMiniGrid](./EnvsTemplates/MyMiniGrid.py)
- [Using Panda Robot](./EnvsTemplates/MyPandaRobot.py)
- [Using gymnasium](./EnvsTemplates/gymnasium-basic.py)
- [Using gymnasium-robotics](./EnvsTemplates/gymnasium-robotics.py)
- [Using MyFetchRobot](./EnvsTemplates/MyFetchRobot.py)
- [Using MyMujoco](./EnvsTemplates/MyMujoco.py)

## Algorithms

- [x] [DQN](./RLAlgos/DQN.py)
- [x] [NoisyNet-DQN](./RLAlgos/DQN.py)
- [x] [DDPG](./RLAlgos/DDPG.py)
- [x] [TD3](./RLAlgos/TD3.py)
- [ ] PPG
- [x] [SAC](./RLAlgos/SAC.py)
- [x] [PPO](./RLAlgos/PPO.py)
- [x] [RPO](./RLAlgos/PPO.py)
- [x] [RND](./RLAlgos/RND.py)
- [ ] QDagger

### Setup

- If you are using the `gymnasium>=0.29.1` package, to render the mujoco, robotics, etc. environments, you need to modify the `site-packages\gymnasium\envs\mujoco\mujoco_rendering.py` file: replace the `solver_iter` (in around line 593) to `solver_niter`.
- If you don't use PyCharm, need to execute: `export PYTHONPATH=<path to RLEnvsAlgos>:$PYTHONPATH`.

### Running Scripts

- DQN algorithm (can only work on discrete action spaces):
    * [DQN in classic control](./run-scripts/dqn.py)
    * [DQN in Atari games](./run-scripts/dqn-atari.py)
    * [DQN in MiniGrid environments](./run-scripts/dqn-minigrid.py)
- PPO algorithm:
    * [PPO in classic control](./run-scripts/ppo.py)
    * [PPO in Atari games](./run-scripts/ppo-atari.py)
    * [PPO in continuous control (e.g., Mujoco)](./run-scripts/ppo-continuous.py)
    * [PPO in robotics control (e.g., gymnasium-robotics, MyFetchRobot)](./run-scripts/ppo-robotics.py)
- RPO algorithm (can only work on continuous action spaces):
    * [RPO in continuous control](./run-scripts/rpo.py)
- RND algorithm:
    * [RND in continuous control](./run-scripts/rnd-continuous.py)
    * [RND in robotics control (e.g., gymnasium-robotics, MyFetchRobot)](./run-scripts/rnd-robotics.py)
    * [RND in MiniGrid environments](./run-scripts/rnd-minigrid.py)
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