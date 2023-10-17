"""
The script to run SAC on classic control environments.
"""

import argparse

from RLAlgos.SAC import SAC

from Networks.ActorNetworks import SACActor
from Networks.QValueNetworks import QNetworkContinuousControl


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAC on classic control environments.")

    parser.add_argument("--exp-name", type=str, default="sac")

    parser.add_argument("--env-id", type=str, default="Ant-v4")
    parser.add_argument("--render", type=bool, default=False)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--rb-optimize-memory", type=bool, default=False)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--q-lr", type=float, default=1e-3)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--target-network-frequency", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.005)

    parser.add_argument("--policy-frequency", type=int, default=2)

    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--alpha-autotune", type=bool, default=True)

    parser.add_argument("--write-frequency", type=int, default=100)
    parser.add_argument("--save-folder", type=str, default="./sac/")

    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--learning-starts", type=int, default=5e3)

    args = parser.parse_args()
    return args


def run():
    args = parse_args()

    agent = SAC(
        env_id=args.env_id,
        actor_class=SACActor,
        critic_class=QNetworkClassicControl,
        exp_name=args.exp_name,
        render=args.render,
        seed=args.seed,
        cuda=args.cuda,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        rb_optimize_memory=args.rb_optimize_memory,
        batch_size=args.batch_size,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        alpha_lr=args.alpha_lr,
        target_network_frequency=args.target_network_frequency,
        tau=args.tau,
        policy_frequency=args.policy_frequency,
        alpha=args.alpha,
        alpha_autotune=args.alpha_autotune,
        write_frequency=args.write_frequency,
        save_folder=args.save_folder
    )

    agent.learn(total_timesteps=args.total_timesteps, learning_starts=args.learning_starts)

    agent.save(indicator="final")


if __name__ == "__main__":
    run()
