"""Load and run policy"""

import argparse
import os
import shutil
from functools import partial

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import (VecNormalize,
                                              sync_envs_normalization,
                                              VecFrameStack,
                                              VecTransposeImage)
import rlhp.general_utils as utils
import wandb

from gym.wrappers import GrayScaleObservation, ResizeObservation, Monitor
from rlhp.wrappers import PixelObservationWrapperCustom, RewardModelWrapper

USER="usman391"

def load_config(d):
    config = utils.load_dict_from_json(d, "config")
    config = utils.dict_to_namespace(config)
    return config

def run_policy(args):
    # Configure paths (restore from W&B server if needed)
    print("xxxxxxx")
    if args.remote:
        # Save everything in wandb/remote/<run_id>
        load_dir = os.path.join("./wandb/remote/", args.load_dir.split('/')[-1])
        utils.del_and_make(load_dir)
        # Restore form W&B
        wandb.init(dir=load_dir)
        print(load_dir)
        #run_path = os.path.join(USER, args.load_dir)
        run_path = args.load_dir
        wandb.restore("requirements.txt", run_path=run_path, root=load_dir)
        wandb.restore("config.json", run_path=run_path, root=load_dir)
        config = load_config(load_dir)
        if not config.dont_normalize_obs:
            wandb.restore("train_env_stats.pkl", run_path=run_path, root=load_dir)
        wandb.restore("best_model.zip", run_path=run_path, root=load_dir)
    else:
        load_dir = os.path.join(args.load_dir, "files")
        config = load_config(load_dir)

    save_dir = os.path.join(load_dir, args.save_dir)
    utils.del_and_make(save_dir)
    model_path = os.path.join(load_dir, "best_model")

    # Load model
    if config.rl_algo == 'ppo':
        model = PPO.load(model_path)
    elif config.rl_algo == 'sac':
        model = SAC.load(model_path)

    # Create env, model
    def make_env():
        env_id = args.env_id or config.eval_env_id
        env_wrappers, vec_wrappers = None, None
        if config.use_pixel_observations:
            env_wrappers = [partial(PixelObservationWrapperCustom, pixels_only=True),
                            partial(ResizeObservation, shape=84),
                            partial(GrayScaleObservation, keep_dim=True)
                            ]
            vec_wrappers = [
                            partial(VecFrameStack, n_stack=4),
                            VecTransposeImage,
                            ]

        env = utils.make_eval_env(env_id,
                              normalize_obs=False,
                              env_wrappers=env_wrappers,
                              vec_wrappers=vec_wrappers)

        # Restore enviroment stats
        if not config.dont_normalize_obs:
            env = VecNormalize.load(os.path.join(load_dir, "train_env_stats.pkl"), env)
            env.norm_reward = False
            env.training = False

        return env

    # Evaluate and make video
    if not args.dont_make_video:
        env = make_env()
        save_dir = "./"
        utils.eval_and_make_video(env, model, save_dir, "video", args.n_rollouts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", "-l", type=str, default="icrl/wandb/latest-run/")
    parser.add_argument("--is_icrl", "-ii", action='store_true')
    parser.add_argument("--remote", "-r", action="store_true")
    parser.add_argument("--save_dir", "-s", type=str, default="run_policy")
    parser.add_argument("--env_id", "-e", type=str, default=None)
    parser.add_argument("--load_itr", "-li", type=int, default=None)
    parser.add_argument("--n_rollouts", "-nr", type=int, default=3)
    parser.add_argument("--dont_make_video", "-dmv", action="store_true")
    parser.add_argument("--dont_save_trajs", "-dst", action="store_true")
    parser.add_argument("--reward_threshold", "-rt", type=float, default=None)
    parser.add_argument("--length_threshold", "-lt", type=int, default=None)
    args = parser.parse_args()

    run_policy(args)


if __name__ == '__main__':
    main()

