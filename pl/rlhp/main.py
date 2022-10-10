from copy import deepcopy
import argparse
import importlib
import json
import os
import pickle
import sys
import time
from functools import partial
import gym
import numpy as np
import stable_baselines3.common.callbacks as callbacks
from stable_baselines3 import PPO
from stable_baselines3.common import logger
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import (VecNormalize,
                                              sync_envs_normalization,
                                              VecFrameStack,
                                              VecTransposeImage)

import rlhp.general_utils as utils
import wandb
from rlhp.rlfhp import TrainRMfPrefsCallback, RewardModelCallback
from rlhp.utils.reward_nets import setup_reward_model_and_cb
from rlhp.utils.policy_setup import setup_policy
from gym.wrappers import GrayScaleObservation, ResizeObservation, Monitor
from rlhp.wrappers import PixelObservationWrapperCustom, RewardModelWrapper

def rlfhp(config):
    env_wrappers, vec_wrappers = None, None
    # In case we want to use pixel based observations
    if config.use_pixel_observations:
        env_wrappers = [partial(PixelObservationWrapperCustom, pixels_only=True),
                        partial(ResizeObservation, shape=84),
                        partial(GrayScaleObservation, keep_dim=True)
                        ]
        vec_wrappers = [
                        partial(VecFrameStack, n_stack=4),
                        VecTransposeImage,
                        ]


    # Create the vectorized environments
    train_env = utils.make_train_env(env_id=config.train_env_id,
                                     save_dir=config.save_dir,
                                     base_seed=config.seed,
                                     num_threads=\
                                        config.num_threads if config.rl_algo == 'ppo' else 1,
                                     normalize_obs=not config.dont_normalize_obs,
                                     normalize_reward= \
                                        (False if (not config.dont_use_reward_model)
                                            else not config.dont_normalize_reward),
                                     gamma=config.gamma,
                                     env_wrappers=env_wrappers,
                                     vec_wrappers=vec_wrappers,
                                     )

    # TODO: should we normalize obs here or not?
    make_sampling_env_fn = lambda : utils.make_eval_env(env_id=config.train_env_id,
                                               normalize_obs=False,
                                               env_wrappers=env_wrappers,
                                               vec_wrappers=vec_wrappers)


    eval_env = utils.make_eval_env(env_id=config.eval_env_id,
                                   normalize_obs=not config.dont_normalize_obs,
                                   env_wrappers=env_wrappers,
                                   vec_wrappers=vec_wrappers)

    if not config.skip_video and config.device == 'cuda':
        env_wrappers_eval = [] if env_wrappers is None else deepcopy(env_wrappers)
        env_wrappers_eval.append(partial(Monitor, directory=os.path.join(config.save_dir, "vids"), 
                                    video_callable=lambda x:True))
        eval_env_for_video = utils.make_eval_env(env_id=config.eval_env_id,
                                       normalize_obs=not config.dont_normalize_obs,
                                       env_wrappers=env_wrappers_eval,
                                       vec_wrappers=vec_wrappers)
    else:
        eval_env_for_video = None
        
    reward_model, reward_model_cb = setup_reward_model_and_cb(train_env, config)
    if reward_model is not None:
        logger.log(reward_model)
        env_wrappers_eval2 = [] if env_wrappers is None else deepcopy(env_wrappers)
        env_wrappers_eval2 += [partial(RewardModelWrapper, reward_model=reward_model)]
        eval_env_with_rm_wrapper = utils.make_eval_env(
                                       env_id=config.eval_env_id,
                                       normalize_obs=not config.dont_normalize_obs,
                                       env_wrappers=env_wrappers_eval2,
                                       vec_wrappers=vec_wrappers)

    # Set specs
    is_discrete = isinstance(train_env.action_space, gym.spaces.Discrete)
    obs_dim = train_env.observation_space.shape[0]
    acs_dim = train_env.action_space.n if is_discrete else train_env.action_space.shape[0]

    action_low, action_high = None, None
    if isinstance(train_env.action_space, gym.spaces.Box):
        action_low, action_high = train_env.action_space.low, train_env.action_space.high

    # Logger
    sb_logger = logger.HumanOutputFormat(sys.stdout)

    if not config.dont_train_reward_model and reward_model is not None:
        rlhp_callback = TrainRMfPrefsCallback(
                rl_algo           = config.rl_algo,
                total_comparisons = config.total_comparisons,
                total_timesteps   = config.timesteps,
                init_comparisons_pct \
                                  = config.init_comparisons_pct,
                use_demos         = config.use_demos_to_generate_prefs,
                gathering_counts = config.dummy_human_gathering_counts,
                transition_oversampling \
                                  = config.transition_oversampling,
                seed              = config.seed,
                make_sampling_env_fn \
                                  = make_sampling_env_fn,
                parallel_data_collection \
                                  = False,
                allow_variable_horizon \
                                 = (not config.dont_allow_variable_horizon),
                fragment_length  = config.fragment_length,
                temperature      = config.dummy_human_temp,
                discount_factor  = config.dummy_human_gamma, #config.reward_gamma,
                return_prob      = config.dummy_human_return_prob,
                model            = reward_model,
                noise_prob       = config.rm_noise_prob,
                weight_decay     = config.rm_weight_decay,
                optimizer_type   = 'AdamW',
                epochs           = config.rm_epochs,
                batch_size       = config.rm_batch_size,
                learning_rate    = config.rm_learning_rate,
                anneal_old_data  = config.rm_anneal_old_data,
                initial_epoch_multiplier \
                                 = config.rm_initial_epoch_multiplier,
                save_all_reward_models = config.save_all_reward_models,
                disable_all_saving = config.disable_all_saving,
                save_dir         = config.save_dir,
                data_dir         = config.data_dir,
                device           = config.device
                        )

    # order of callbacks is important
    # we want to call reward_model_cb after rlhp_train_cb
    if config.dont_use_reward_model:
        all_callbacks = []
    elif config.dont_train_reward_model:
        if config.rl_algo in ['ppo', 'trpo']:
            all_callbacks = [reward_model_cb]
        else:
            all_callbacks = []
    elif config.rl_algo in ['ppo', 'trpo']:
        all_callbacks = [rlhp_callback, reward_model_cb] 
    else:
        all_callbacks = [rlhp_callback] 

    # Define and train model
    model = setup_policy(train_env, config, reward_model)

    # All callbacks
    if eval_env_for_video is not None:
        video_callback = callbacks.VideoCallback(
                eval_env_for_video, video_freq=40000,verbose=0,
        )
        all_callbacks.extend([video_callback])

    save_periodically = callbacks.CheckpointCallback(
            config.save_every, os.path.join(config.save_dir, "models"),
            verbose=0
    )
    save_env_stats = utils.SaveEnvStatsCallback(train_env, config.save_dir)
    save_best = callbacks.EvalCallback(
            eval_env, eval_freq=config.eval_every, deterministic=False,
            best_model_save_path=config.save_dir, verbose=0,
            callback_on_new_best=save_env_stats
    )

    # Organize all callbacks in list
    if not config.disable_all_saving:
        all_callbacks.extend([save_periodically,
                              save_best,
                              ])

    if not config.dont_use_reward_model:
        rm_reward_callback = callbacks.EvalCallback(
                eval_env_with_rm_wrapper, eval_freq=config.eval_every, deterministic=False,
                best_model_save_path=None, verbose=0, label_str="RewardModelEval",
                x_str='_',
        )
        all_callbacks.extend([rm_reward_callback])

    if config.save_rollouts or config.save_rollouts_every is not None:
        if config.save_rollouts_every is not None:
            save_rollouts_cb = callbacks.SaveRolloutsEveryNStepsCallback(
                                config.save_rollouts_every,
                                sampling_env = eval_env,
                                k = 10,
                                save_path = config.save_dir,
                                device = config.device
                                )
        else:
            save_rollouts_cb = callbacks.SaveRolloutsComplexCallback(
                                start_saving_at=10000,
                                sampling_env = eval_env,
                                k = 10,
                                save_path = config.save_dir,
                                device = config.device
                                )
        all_callbacks.extend([save_rollouts_cb])

    # Train
    model.learn(total_timesteps=int(config.timesteps),
                callback=all_callbacks)

    # Save normalization stats
    if isinstance(train_env, VecNormalize) and not config.disable_all_saving:
        train_env.save(os.path.join(config.save_dir, "train_env_stats.pkl"))

    # Make video of final model
    if not config.wandb_sweep and not config.skip_video:
        sync_envs_normalization(train_env, eval_env)
        try:
            utils.eval_and_make_video(eval_env, model, config.save_dir, "final_policy")
        except:
            pass

    if config.sync_wandb:
        utils.sync_wandb(config.save_dir, 120)

def main(raw_args=None):
    start = time.time()
    parser = argparse.ArgumentParser()
    # ========================== Setup ============================== #
    parser.add_argument("--config_file", "-cf", type=str, default=None,
        help="You can pass a config file to override argparse defaults.")
    parser.add_argument("--project", "-p", type=str, default="ABC",
        help="Wandb project")
    parser.add_argument("--group", "-g", type=str, default=None,
        help="Wandb Group")
    parser.add_argument("--name", "-n", type=str, default=None,
        help="Wandb experiment name.")
    parser.add_argument("--device", "-d", type=str, default="cpu")
    parser.add_argument("--verbose", "-v", type=int, default=2,
        help="verbose paramater in SB3.")
    parser.add_argument("--wandb_sweep", "-ws", type=bool, default=False,
        help="Are you running a wandb sweep?")
    parser.add_argument("--sync_wandb", "-sw", action="store_true",
        help="Sync at the end of experiment even if wandb offline.")
    # ======================== Environment ========================== #
    parser.add_argument("--train_env_id", "-tei", type=str, default="HalfCheetah-v3",
        help="Gym environment to train on.")
    parser.add_argument("--eval_env_id", "-eei", type=str, default="HalfCheetah-v3",
        help="Gym environment to evaluate on.")
    parser.add_argument("--dont_normalize_obs", "-dno", action="store_true",
        help="Do not normalize observations using VecNormalize wrapper.")
    parser.add_argument("--dont_normalize_reward", "-dnr", action="store_true",
        help="Do not normalize rewards. This controls both reward model reward \
              and environment reward behaviour.")
    parser.add_argument("--seed", "-s", type=int, default=None)
    # ======================== Networks ============================== #
    parser.add_argument("--rl_algo", "-rla", type=str, default="ppo",
        choices=['ppo', 'sac', 'td3', 'ddpg'], help="Which RL algo to use?")
    parser.add_argument("--policy_name", "-pn", type=str, default="MlpPolicy",
        choices=['MlpPolicy', 'CnnPolicy'],
        help="Let's you use CnnPolicy with pixel envs.")
    parser.add_argument("--shared_layers", "-sl", type=int, default=None, nargs='*',
        help="Only for PPO. Let's you share layers between PPO policy and vf.\
              See SB3 PPO ddcoumentation for more.")
    parser.add_argument("--policy_layers", "-pl", type=int, default=[64,64], nargs='*',
        help="MLP size for policy layers. For CNN policy we use NatureCNN.")
    parser.add_argument("--reward_vf_layers", "-vfl", type=int, default=[64,64], nargs='*',
        help="MLP size for value fucntion.")
    # ========================= Training ============================ #
    parser.add_argument("--timesteps", "-t", type=lambda x: int(float(x)), default=1e6,
        help="How many steps to take in the environment?")
    parser.add_argument("--learning_rate", "-lr", type=float, default=3e-4,
        help="SB3 model learning rate.")
    parser.add_argument("--n_steps", "-ns", type=int, default=2048,
        help="PPO param. See SB3.")
    parser.add_argument("--batch_size", "-bs", type=int, default=64)
    parser.add_argument("--n_epochs", "-ne", type=int, default=10)
    parser.add_argument("--num_threads", "-nt", type=int, default=5)
    parser.add_argument("--save_every", "-se", type=float, default=5e5,
        help="Save after this many timesteps in train_env.")
    parser.add_argument("--eval_every", "-ee", type=float, default=2048,
        help="Evaluate policy in eval_env after this many timesteps.")
    parser.add_argument('--data_dir', '-dd', type=str, default=None,
        help="Path to expert or oracle data.")
    # =========================== MDP =============================== #
    parser.add_argument("--gamma", "-rg", type=float, default=0.99,
        help="Reward discount factor.")
    parser.add_argument("--gae_lambda", "-rgl", type=float, default=0.95,
        help="GAE Lambda value for GAE. See GAE paper for more details.")
    # ========================= PPO Only ============================== #
    # see PPO documentation
    parser.add_argument("--clip_range", "-cr", type=float, default=0.2)
    parser.add_argument("--clip_range_vf", "-crvf", type=float, default=None)
    parser.add_argument("--ent_coef", "-ec", type=float, default=0.)
    parser.add_argument("--vf_coef", "-vfc", type=float, default=0.5)
    parser.add_argument("--target_kl", "-tk", type=float, default=None)
    parser.add_argument("--max_grad_norm", "-mgn", type=float, default=0.5)
    # ========================= SAC Only ============================== #
    # see SB3 documentation
    parser.add_argument("--train_freq", "-tf", type=float, default=1)
    parser.add_argument("--learning_starts", "-ls", type=float, default=10000)
    # Only for TD3
    parser.add_argument("--policy_delay", '-pd', type=int, default=2)
    # ========================= Reward Net ========================== #
    parser.add_argument("--reward_net_type", "-rnt", default="mlp",
        choices=['mlp', 'nature_cnn', 'borja_cnn'],
        help="What kind of NN strucutre to use for reward model?")
    parser.add_argument("--reward_net_layers", "-rl", type=int, default=[30,30], nargs='*',
        help="MLP layers in reward model. This is active for both mlp and cnn\
              type reward models.")
    parser.add_argument("--reward_net_conv_features_dim", "-rcfd", type=int, default=256, 
        help="What size embedding should CNN project pixel observations to?")
    parser.add_argument("--reward_scheme", "-rs", type=str, default="sa",
    help="reward scheme to be used by reward model; must be in \
            ['s', 'ss', 'sa', 'sasd']. s corresponds to state only,\
            sa corresponds to state+action and so on.")
    parser.add_argument('--reward_model_path', '-rmp', type=str, default=None,
        help="Path to a saved reward model.")
    parser.add_argument('--override_dont_train_saved_rm_assertion',
        action="store_true",
        help="By default, if reward_model_path is not None, an assertion is\
              checked that dont_train_reward_model is true. This disables\
              that assertion and lets you train a saved reward model.")
    # ======================= RLfHP ================================= #
    # These parameters control macro level behaviour of RLfHP algo.
    # There are all same as in imitation with one big difference
    # Imitation has parameter 'comparisons_per_iter' which is used
    # with other params like total_comparisons and init_comparisions_pct
    # to determine number of iterations
    # We however have 'iterations' directly as a parameter with the
    # name 'dummy_human_gathering_counts'. This parameter controls
    # the number of data collection + reward model updates round
    # The other big difference from imitation is that this code has
    # capability/plans to allow use of demonstrations
    parser.add_argument('--total_comparisons', '-tc', type=int, default=1e4)
    parser.add_argument('--init_comparisons_pct', '-icpct', type=float, default=0.1)
    parser.add_argument('--dummy_human_gathering_counts', '-gc', type=int, default=10)
    parser.add_argument('--use_demos_to_generate_prefs', '-use_demos',  action='store_true')
    # ============== Sampling for RLfHP ============================= #
    parser.add_argument('--dont_allow_variable_horizon', '-davh', action="store_true")
    parser.add_argument('--transition_oversampling', '-to', type=int, default=2)
    parser.add_argument('--fragment_length', '-fl', type=int, default=50)
    parser.add_argument('--dummy_human_temp', '-dht', type=int, default=50)
    parser.add_argument('--dummy_human_gamma', '-dhg', type=int, default=1.0)
    parser.add_argument('--dummy_human_return_prob', '-dhbp', action="store_true")
    # ============== Training Reward Model ========================== #
    parser.add_argument('--rm_noise_prob', '-rmnp', type=float, default=0.0)
    parser.add_argument('--rm_weight_decay', '-rmwd', type=float, default=0.0)
    parser.add_argument('--rm_optimizer', '-rmo', type=str, default='AdamW')
    parser.add_argument('--rm_epochs', '-rme', type=int, default=20)
    parser.add_argument('--rm_batch_size', '-rmbs', type=int, default=64)
    parser.add_argument('--rm_learning_rate', '-rmlr', type=float, default=3e-4)
    parser.add_argument('--rm_anneal_old_data', '-rmaod', action='store_true',
        help="Not implemented. Anneals old data in RM dataset.")
    parser.add_argument('--rm_initial_epoch_multiplier', '-rmiem', type=float, default=3)
    # ============= Memory Footprint ================================ #
    # By default we save policy every save_every, the best policy and the last reward model
    # Using rm_save_all_reward_models additionally triggers saving of reward model at the
    # end of every training run meaning if there were 10 training runs (gathering counts)
    # to update the reward mdoel, there would be 10 saved reward models.
    # disable_all_saving overrides everything and prompts the script to
    # neither save any reward model nor any policy version
    parser.add_argument('--save_all_reward_models', action='store_true')
    parser.add_argument('--disable_all_saving', action='store_true')
    parser.add_argument('--skip_video', action='store_true')
    # ====================== Useful Macros ========================== #
    parser.add_argument('--dont_use_reward_model', action='store_true',
        help="This only runs rl on environment's actual reward.")
    parser.add_argument('--dont_train_reward_model', action='store_true',
        help="This uses reward model but does not train it.")
    parser.add_argument('--use_pixel_observations', action='store_true')
    parser.add_argument('--save_rollouts', action='store_true',
        help="Save rollouts from the policy at various timesteps.")
    parser.add_argument('--save_rollouts_every', type=int, default=None,
        help="This activates SaveRolloutsEveryNstepsCallback.")
    parser.add_argument('--reward_relabeling', action="store_true",
        help="Enables reward relabeling for off policy algos (SAC, TD3, A2C)")
    # ======================= Expert Data =========================== #
    parser.add_argument('--expert_path', '-ep', type=str, default=None)
    parser.add_argument('--expert_rollouts', '-er', type=int, default=20)

    args = vars(parser.parse_args(raw_args))

    # Get default config
    default_config, mod_name = {}, ''
    if args["config_file"] is not None:
        if args["config_file"].endswith(".py"):
            mod_name = args["config_file"].replace('/', '.').strip(".py")
            default_config = importlib.import_module(mod_name).config
        elif args["config_file"].endswith(".json"):
            default_config = utils.load_dict_from_json(args["config_file"])
        else:
            raise ValueError("Invalid type of config file")

    # Overwrite config file with parameters supplied through parser
    # Order of priority: supplied through command line > specified in config
    # file > default values in parser
    raw_args = [] if raw_args is None else raw_args
    # check if main.py is being called directly or as a module
    if 'main.py' == sys.argv[0].split('/')[-1]:
        raw_args = sys.argv[1:]
    config = utils.merge_configs(default_config, parser, args, raw_args)
    # Choose seed
    if config["seed"] is None:
        config["seed"] = np.random.randint(0,100)

    # Get name by concatenating arguments with non-default values. Default
    # values are either the one specified in config file or in parser (if both
    # are present then the one in config file is prioritized)
    config["name"] = utils.get_name(parser, default_config, config, mod_name)

    # Initialize W&B project
    base_dir = './wandb'
    base_dir = base_dir + '/' + config['project'] if config['project'] is not None else base_dir
    base_dir = base_dir + '/' + config['group'] if config['group'] is not None else base_dir
    os.makedirs(base_dir, exist_ok=True)
    run = wandb.init(project=config["project"], name=config["name"], config=config, dir=base_dir,
               group=config['group'], reinit=True, monitor_gym=True)
    wandb.config.save_dir = wandb.run.dir
    config = wandb.config

    print(utils.colorize("Configured folder %s for saving" % config.save_dir,
          color="green", bold=True))
    print(utils.colorize("Name: %s" % config.name, color="green", bold=True))

    # Save config
    utils.save_dict_as_json(config.as_dict(), config.save_dir, "config")

    # Train
    rlfhp(config)

    end = time.time()
    print(utils.colorize("Time taken: %05.2f minutes" % ((end-start)/60),
          color="green", bold=True))

    run.finish()
    return config.save_dir

if __name__ == '__main__':
    main()
