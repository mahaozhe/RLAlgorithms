"""
The script to DDPG on continuous control environments.
"""

import argparse

from RLAlgos.DDPG import DDPG

from Networks.ActorNetworks import DeterministicActorClassicControl
from Networks.QValueNetworks import QNetworkClassicControl


def parse_args():
    parser = argparse.ArgumentParser(description="Run DDPG on continuous control environments.")

    parser.add_argument("--exp-name", type=str, default="ddpg")

    parser.add_argument("--env-id", type=str, default="Ant-v4")
    parser.add_argument("--render", type=bool, default=False)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--rb-optimize-memory", type=bool, default=False)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)

    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.005)

    parser.add_argument("--policy-frequency", type=int, default=2)

    parser.add_argument("--write-frequency", type=int, default=100)
    parser.add_argument("--save-folder", type=str, default="./ddpg/")

    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--learning-starts", type=int, default=25e3)

    args = parser.parse_args()
    return args


def run():
    args = parse_args()

    agent = DDPG(
        env_id=args.env_id,
        actor_class=DeterministicActorClassicControl,
        critic_class=QNetworkClassicControl,
        exp_name=args.exp_name,
        render=args.render,
        seed=args.seed,
        cuda=args.cuda,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        rb_optimize_memory=args.rb_optimize_memory,
        exploration_noise=args.exploration_noise,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        batch_size=args.batch_size,
        tau=args.tau,
        policy_frequency=args.policy_frequency,
        write_frequency=args.write_frequency,
        save_folder=args.save_folder
    )

    agent.learn(total_timesteps=args.total_timesteps, learning_starts=args.learning_starts)

    agent.save(indicator="final")


if __name__ == "__main__":
    run()
