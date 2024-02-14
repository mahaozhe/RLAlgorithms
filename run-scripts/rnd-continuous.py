"""
The script to run RND on my-robotics and my-mujoco environments.
"""

import argparse

from RLAlgos.RND import RND

from Networks.CombinedActorCriticNetworks import RNDMujocoAgent, RNDMujocoModel

from utils.env_makers import sync_vector_mujoco_envs_maker


def parse_args():
    parser = argparse.ArgumentParser(description="Run RND on classic control environments.")

    parser.add_argument("--exp-name", type=str, default="rnd-continuous")

    parser.add_argument("--env-id", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--num-envs", type=int, default=1)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)

    parser.add_argument("--rollout-length", type=int, default=2048)
    parser.add_argument("--num-mini-batches", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=10)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--anneal-lr", type=bool, default=True)

    parser.add_argument("--norm-adv", type=bool, default=True)
    parser.add_argument("--clip-value-loss", type=bool, default=True)

    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)

    parser.add_argument("--write-frequency", type=int, default=100)
    parser.add_argument("--save-folder", type=str, default="./rnd-continuous/")

    parser.add_argument("--total-timesteps", type=int, default=1000000)

    args = parser.parse_args()
    return args


def run():
    args = parse_args()

    # ! note the env maker needs the additional argument gamma
    envs = sync_vector_mujoco_envs_maker(env_id=args.env_id, num_envs=args.num_envs, seed=args.seed, gamma=args.gamma)

    agent = RND(envs=envs, agent_class=RNDMujocoAgent, rn_class=RNDMujocoModel, exp_name=args.exp_name, seed=args.seed,
                cuda=args.cuda, gamma=args.gamma, gae_lambda=args.gae_lambda, rollout_length=args.rollout_length,
                lr=args.lr, eps=args.eps, anneal_lr=args.anneal_lr, num_mini_batches=args.num_mini_batches,
                update_epochs=args.update_epochs, norm_adv=args.norm_adv, clip_value_loss=args.clip_value_loss,
                clip_coef=args.clip_coef, entropy_coef=args.entropy_coef, value_coef=args.value_coef,
                max_grad_norm=args.max_grad_norm, target_kl=args.target_kl, write_frequency=args.write_frequency,
                save_folder=args.save_folder)

    agent.learn(total_timesteps=args.total_timesteps)

    agent.save(indicator="final")


if __name__ == "__main__":
    run()
