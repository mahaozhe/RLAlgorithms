"""
The script to run NoisyNet-DQN on MiniGrid environments.
"""

import argparse

from RLAlgos.DQN import NoisyNetDQN

from Networks.NoisyNetworks import NoisyQNetMiniGrid

from utils.env_makers import minigrid_env_maker


def parse_args():
    parser = argparse.ArgumentParser(description="Run NoisyNet-DQN on MiniGrid environments.")

    parser.add_argument("--exp-name", type=str, default="noisy-net-dqn-minigrid")

    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--render", type=bool, default=False)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--rb-optimize-memory", type=bool, default=False)
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--target-network-frequency", type=int, default=500)
    parser.add_argument("--tau", type=float, default=1.)

    parser.add_argument("--train-frequency", type=int, default=10)

    parser.add_argument("--write-frequency", type=int, default=100)
    parser.add_argument("--save-folder", type=str, default="./noisy-net-dqn-minigrid/")

    parser.add_argument("--total-timesteps", type=int, default=500000)
    parser.add_argument("--learning-starts", type=int, default=10000)

    args = parser.parse_args()
    return args


def run():
    args = parse_args()

    env = minigrid_env_maker(env_id=args.env_id, seed=args.seed, render=args.render)

    agent = NoisyNetDQN(env=env, noisy_q_network_class=NoisyQNetMiniGrid, exp_name=args.exp_name, seed=args.seed,
                        cuda=args.cuda, learning_rate=args.learning_rate, buffer_size=args.buffer_size,
                        rb_optimize_memory=args.rb_optimize_memory, gamma=args.gamma, tau=args.tau,
                        target_network_frequency=args.target_network_frequency, batch_size=args.batch_size,
                        train_frequency=args.train_frequency, write_frequency=args.write_frequency,
                        save_folder=args.save_folder)

    agent.learn(total_timesteps=args.total_timesteps, learning_starts=args.learning_starts)

    agent.save(indicator="final")


if __name__ == "__main__":
    run()
