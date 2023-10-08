"""
The script to run DQN on Atari environments.
"""

import argparse

import gymnasium as gym

from RLAlgos.DQN import DQN_Atari

from Networks.QNetworks import QNetClassicControl


def parse_args():
    parser = argparse.ArgumentParser(description="Run DQN on Atari environments.")

    parser.add_argument("--exp-name", type=str, default="dqn-atari")

    parser.add_argument("--env-id", type=str, default="ALE/MsPacman-v5")
    parser.add_argument("--render", type=bool, default=False)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--rb-optimize-memory", type=bool, default=True)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--target-network-frequency", type=int, default=1000)
    parser.add_argument("--tau", type=float, default=1.)

    parser.add_argument("--start-e", type=float, default=1.0)
    parser.add_argument("--end-e", type=float, default=0.01)
    parser.add_argument("--exploration-fraction", type=float, default=0.1)

    parser.add_argument("--train-frequency", type=int, default=4)

    parser.add_argument("--write-frequency", type=int, default=100)
    parser.add_argument("--save-folder", type=str, default="./dqn-atari/")

    parser.add_argument("--total-timesteps", type=int, default=10000000)
    parser.add_argument("--learning-starts", type=int, default=80000)

    args = parser.parse_args()
    return args


def run():
    args = parse_args()

    agent = DQN_Atari(
        env_id=args.env_id,
        q_network_class=QNetClassicControl,
        exp_name=args.exp_name,
        render=args.render,
        seed=args.seed,
        cuda=args.cuda,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        rb_optimize_memory=args.rb_optimize_memory,
        gamma=args.gamma,
        tau=args.tau,
        target_network_frequency=args.target_network_frequency,
        batch_size=args.batch_size,
        start_e=args.start_e,
        end_e=args.end_e,
        exploration_fraction=args.exploration_fraction,
        train_frequency=args.train_frequency,
        write_frequency=args.write_frequency,
        save_folder=args.save_folder
    )

    agent.learn(total_timesteps=args.total_timesteps, learning_starts=args.learning_starts)

    agent.save(indicator="final")
