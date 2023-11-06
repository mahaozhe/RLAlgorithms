# RL Experimental Environments

## Environments

All RL environments packages can be found in [RLEnvs](./RLEnvs/) package.

- [MyMiniGrid](./RLEnvs/MyMiniGrid)
- [Panda Robot](./RLEnvs/MyPandaRobot)
- [gymnasium](./RLEnvs/gymnasium)

### Templates

- [Using MyMiniGrid](./EnvsTemplates/MyMiniGrid.py)
- [Using Panda Robot](./EnvsTemplates/MyPandaRobot.py)
- [Using gymnasium](./EnvsTemplates/gymnasium-basic.py)

## Algorithms

- [x] [DQN](./RLAlgos/DQN.py)
- [x] [DDPG](./RLAlgos/DDPG.py)
- [x] [TD3](./RLAlgos/TD3.py)
- [ ] PPG
- [x] [SAC](./RLAlgos/SAC.py)
- [x] [PPO](./RLAlgos/PPO.py)
- [x] [RPO](./RLAlgos/PPO.py)
- [ ] RND
- [ ] QDagger

### Setup

- If you are using the `gymnasium>=0.29.1` package, to render the mujoco, robotics, etc. environments, you need to modify the `site-packages\gymnasium\envs\mujoco\mujoco_rendering.py` file: replace the `solver_iter` (in around line 593) to `solver_niter`.
- If you don't use PyCharm, need to execute: `export PYTHONPATH=<path to RLEnvsAlgos>:$PYTHONPATH`.

### Running Scripts

- DQN algorithm (can only work on discrete action spaces):
    * [DQN in classic control](./run-scripts/dqn.py)
    * [DQN in Atari games](./run-scripts/dqn-atari.py)
- PPO algorithm:
    * [PPO in classic control](./run-scripts/ppo.py)
    * [PPO in Atari games](./run-scripts/ppo-atari.py)
    * [PPO in continuous control (e.g., Mujoco)](./run-scripts/ppo-continuous.py)
- RPO algorithm (can only work on continuous action spaces):
    * [RPO in continuous control](./run-scripts/rpo.py)
- DDPG algorithm (can only work on continuous action spaces):
    * [DDPG in continuous control](./run-scripts/ddpg.py)
- TD3 algorithm (can only work on continuous action spaces):
    * [TD3 in continuous control](./run-scripts/td3.py)
- SAC algorithm:
    * [SAC in continuous control](./run-scripts/sac.py)
    * [SAC in Atari games](./run-scripts/sac-atari.py)