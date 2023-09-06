"""
The Deep Q Network (DQN) algorithm.

* Do not support the vectorized environment.

references:
- cleanrl: https://github.com/vwxyzjn/cleanrl/tree/master
- original paper: https://www.nature.com/articles/nature14236

! Note: the code is completed with the help of copilot.
"""

import random
import torch
import numpy as np


class DQN:
    """
    The DQN base algorithm.
    """

    def __init__(self, env_id, agent_class, exp_name="test", render=False, seed=1, cuda=0, learning_rate=2.5e-4,
                 buffer_size=10000, gamma=0.99, tau=1., target_network_frequency=500, batch_size=128, start_e=1,
                 end_e=0.05, exploration_fraction=0.5, train_frequency=10, write_frequency=100, save_folder="./runs/"):
        """
        Initialize the DQN algorithm.
        :param env_id: the environment id
        :param agent_class: the agent class
        :param exp_name: the experiment name
        :param render: whether to render the environment
        :param seed: the random seed
        :param cuda: whether to use cuda
        :param learning_rate: the learning rate
        :param buffer_size: the replay memory buffer size
        :param gamma: the discount factor gamma
        :param tau: the target network update rate
        :param target_network_frequency: the timesteps it takes to update the target network
        :param batch_size: the batch size of sample from the reply memory
        :param start_e: the starting epsilon for exploration
        :param end_e: the ending epsilon for exploration
        :param exploration_fraction: the fraction of `total-timesteps` it takes from start-e to go end-e
        :param train_frequency: the frequency of training
        :param write_frequency: the frequency of writing to tensorboard
        :param save_folder: the folder to save the model
        """

        self.seed = seed

        # set the random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

        self.env = self.make_env(env_id, seed)
