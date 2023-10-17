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
- [ ] C51
- [x] [DDPG](./RLAlgos/DDPG.py)
- [x] TD3
- [x] [SAC](./RLAlgos/SAC.py)
- [ ] TRPO?
- [ ] PPO
- [ ] RPO

### Running Scripts

- DQN algorithm (can only work on discrete action spaces):
    * [DQN in classic control](./run-scripts/dqn.py)
    * [DQN in Atari games](./run-scripts/dqn-atari.py)
- SAC algorithm:
    * [SAC in continuous control](./run-scripts/sac.py)
    * [SAC in Atari games](./run-scripts/sac-atari.py)
- DDPG algorithm (can only work on continuous action spaces):
    * [DDPG in continuous control](./run-scripts/ddpg.py)
- TD3 algorithm (can only work on continuous action spaces):
    * [TD3 in continuous control](./run-scripts/td3.py)