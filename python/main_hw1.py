import argparse
from learning_algorithms import PGTrainer
from utils import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '-e', type=str, default='LunarLander-v2')
    parser.add_argument('--rng_seed', '-rng', default=6369)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true', help='if present, reward-to-go will be applied')
    parser.add_argument('--reward_discount', '-rd', action='store_true', help='if present, reward-discounting will be applied')
    parser.add_argument('--n_rollout', '-nr', type=int, default=10, help='number of rollouts played (and trained) in total')
    parser.add_argument('--n_trajectory_per_rollout', '-ntr', type=int, default=2, help='number of trajectories (episodes) per rollout to be gathered')
    parser.add_argument('--hidden_dim', '-hdim', type=int, default=64, help='hidden dimension of the policy-net')
    parser.add_argument('--lr', '-lr', type=float, default=3e-3, help='learning rate')
    parser.add_argument('--exp_name', '-xn', type=str, default='my_exp', help='name of the experiment')
    args = parser.parse_args()
    params = vars(args)

    seed_everything(params['rng_seed'])

    trainer = PGTrainer(params)
    trainer.run_training_loop()


if __name__ == '__main__':
    main()
