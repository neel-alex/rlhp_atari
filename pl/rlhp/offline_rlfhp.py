from __future__ import annotations  # for backward compatibility
import os, time, wandb, argparse, sys
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import random, math
from typing import Optional, Callable, Tuple
from functools import partial

from scipy import special
from scipy.stats import kendalltau
from scipy.spatial.distance import hamming

from stable_baselines3.common import logger

from rlhp.rlfhp import (FragmentsGenerator,
                        SyntheticGatherer,
                        PreferenceDataset,
                        flatten_trajectories,
                        preference_collate_fn)
import rlhp.general_utils as utils
from rlhp.utils.reward_nets import setup_reward_model_and_cb
from gym.wrappers import GrayScaleObservation, ResizeObservation, Monitor
from rlhp.wrappers import PixelObservationWrapperCustom, RewardModelWrapper


class WandbLogger:
    def __init__(self):
        self.log_dict = {}

    def record(self, k, v):
        self.log_dict[k] = v

    def dump(self, step):
        wandb.log(self.log_dict, step=step)
        self.log_dict = {}

class PreferenceDatasetOffline(th.utils.data.Dataset):
    def __init__(self):
        self.fragments1 = []
        self.fragments2 = []
        self.preferences = np.array([], dtype=np.float32)

    def push(self, fragments, preferences):
        fragments1, fragments2 = zip(*fragments)
        if preferences.shape != (len(fragments), ):
            raise ValueError(
                f"Unexpected preferences shape {preferences.shape}, "
                f"expected {(len(fragments), )}",
            )
        if preferences.dtype != np.float32:
            raise ValueError("preferences should have dtype float32")

        self.fragments1.extend(fragments1)
        self.fragments2.extend(fragments2)
        self.preferences = np.concatenate((self.preferences, preferences))

    def process(self, reward_model):
        self.preferences = th.from_numpy(self.preferences)
        self.obs1, self.acs1, self.next_obs1, self.dones1 = [], [], [], []
        self.obs2, self.acs2, self.next_obs2, self.dones2 = [], [], [], []
        for frag1, frag2 in zip(self.fragments1, self.fragments2):
            t1 = flatten_trajectories(frag1)
            state_th, action_th, next_state_th, done_th = \
                    reward_model.preprocess(t1['obs'], t1['acs'], t1['next_obs'],
                                            t1['dones'])
            self.obs1.append(state_th)
            self.acs1.append(action_th)
            self.next_obs1.append(next_state_th)
            self.dones1.append(done_th)

            t2 = flatten_trajectories(frag2)
            state_th, action_th, next_state_th, done_th = \
                    reward_model.preprocess(t2['obs'], t2['acs'], t2['next_obs'],
                                            t2['dones'])
            self.obs2.append(state_th)
            self.acs2.append(action_th)
            self.next_obs2.append(next_state_th)
            self.dones2.append(done_th)


    def __getitem__(self, i):
        return (self.obs1[i], self.acs1[i], self.next_obs1[i], self.dones1[i]),\
               (self.obs2[i], self.acs2[i], self.next_obs2[i], self.dones2[i]),\
               self.preferences[i]

    def __len__(self) -> int:
        assert len(self.fragments1) == len(self.fragments2) == len(self.preferences)
        return len(self.fragments1)

    def save(self, path) -> None:
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path) -> "PreferenceDataset":
        with open(path, "rb") as file:
            return pickle.load(file)


class TrainRMfPrefsOffline:
    """
    This callback trains a reward model from binary preferences collected
    from a (dummy) human on trajectories collected from  an agent acting
    in the environment.
    """
    def __init__(
        self,
        seed: int,
        # data collection parameters
        total_comparisons: int,
        ### Fragmenter
        fragment_length: int,
        ### Preference Collector
        temperature: float,
        discount_factor: float,
        return_prob: bool,
        # train
        model: Callable,
        noise_prob: float,
        weight_decay: float,
        optimizer_type: str, # Adam, AdamW
        epochs: int,
        batch_size: int,
        learning_rate: int,
        save_all_reward_models: bool,
        disable_all_saving: bool,
        save_every: int,
        plot_every: int,
        eval_every: int,
        save_dir: str,
        data_dir: str,
        use_data_from: str,
        num_val_agents: int,
        threshold: int = 50,
        device: str = 'cpu'):
        """
        See docstring for TrainRMfromPrefsCallback in rlhp/rlfhp.py
        """
        self.total_comparisons = total_comparisons
        self.discount_factor = discount_factor
        self.threshold = threshold
        self.temperature = temperature
        self.noise_prob = noise_prob
        self.device = device
        self.save_every = save_every
        self.plot_every = plot_every
        self.eval_every = eval_every
        self.use_data_from = use_data_from
        self.num_val_agents = num_val_agents
        self.logger = WandbLogger()

        # initialize objects to simulate preference collection

        ## Fragments Generator
        self.frag_gen = FragmentsGenerator(fragment_length, seed)
        self.fragment_length = fragment_length

        ## Preference Collection
        self.dummy_human = SyntheticGatherer(
                        temperature=temperature,
                        discount_factor=discount_factor,
                        sample=not return_prob,
                        custom_logger=self.logger)

        ## Making a dataset
        self.dataset = PreferenceDatasetOffline()

        # setup training
        self.reward_model = model
        self.batch_size = batch_size
        self.epochs = epochs
        if optimizer_type == 'AdamW':
            self.optim = th.optim.AdamW(
                             self.reward_model.parameters(),
                             lr=learning_rate,
                             weight_decay=weight_decay,)
        else:
            raise NotImplementedError

        self.save_all_reward_models = save_all_reward_models
        self.disable_all_saving = disable_all_saving
        self.save_dir = os.path.join(save_dir, 'reward_models')
        os.mkdir(self.save_dir)

        # Load data if needed
        self.trajs = {}
        if data_dir is not None:
            self._load_trajs(data_dir)

        # Some variables for bookkeeping
        self.i = 0 # counter to check which rl_iter we are on
        self.total_comparisons_so_far = 0
        self.total_rm_updates_so_far = 0

    def _prepare_data(self):
        """
        A utility function which samples trajectories from the environment,
        fragments them, have dummy human annotate them and then adds them
        to the dataset.
        """
        for traj in self.train_trajs:
            if traj['dones'][-1]:
                traj['terminal'] = True
            else:
                traj['terminal'] = False
        print(self.total_comparisons)
        frags = self.frag_gen(self.train_trajs, int(self.total_comparisons))
        preferences = self.dummy_human(frags)
        self.dataset.push(frags, preferences)
        self.dataset.process(self.reward_model)
        self.dataloader = th.utils.data.DataLoader(
                            self.dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            pin_memory=True,
       #                     collate_fn=preference_collate_fn,
                        )

    def train(self):
        self._prepare_data()
        self.i = 0
        self._compare_with_orcale_rank(plot=True)
        for _ in tqdm(range(self.epochs)):
            self._train(epochs=1)
            self.i += 1
            if self.i % self.eval_every == 0:
                self._compare_with_orcale_rank(plot = (self.i % self.plot_every == 0))

            if self.i % self.save_every == 0:
                self.save()

            self.logger.record("RM Training/Iteration", self.i)
            self.logger.record("RM Training/Total Comparisons",
                                    self.total_comparisons)

            self.logger.dump(step=self.i)

        # saving
        self.save()

    def _loss(self, trans1, trans2, preferences: np.ndarray) -> th.tensor:
        rews1 = self.reward_model.forward_offline(trans1[0], trans1[1], trans1[2], trans1[3])
        rews2 = self.reward_model.forward_offline(trans2[0], trans2[1], trans2[2], trans2[3])
        probs = self._probability(rews1, rews2)
        predictions = (probs > 0.5).float()
        ground_truth = (preferences > 0.5).float()
        accuracy = (predictions == ground_truth).float().mean()
        return (th.nn.functional.binary_cross_entropy(probs, preferences),
                accuracy.item())

    def _probability(self, rews1: th.tensor, rews2: th.tensor) -> th.tensor:
        if self.discount_factor == 1:
            returns_diff = (rews2 - rews1).sum(dim=1)
        else: 
            discounts = self.discount_factor ** th.arange(rews1.shape[-1], device=self.device)
            discounts = th.stack([discounts] * rews1.shape[0]) 
            returns_diff = (discounts * (rews2 - rews1)).sum(dim=1)
        # clip to avoid overflows
        returns_diff = th.clip(returns_diff, -self.threshold, self.threshold)
        model_probability = (1 / (1 + returns_diff.exp()))
        return self.noise_prob * 0.5 + (1 - self.noise_prob)*model_probability

    def _train(self, epochs):
        for _ in (range(epochs)):
            losses, accuracies = [], []
            for fragment1, fragment2, preferences in self.dataloader:
                self.optim.zero_grad()
                loss, accuracy = self._loss(fragment1, fragment2, preferences)
                loss.backward()
                self.optim.step()
                losses.append(loss.item())
                accuracies.append(accuracy)
            self.logger.record("RM Training/Loss", np.mean(losses))
            self.logger.record("RM Training/Accuracy", np.mean(accuracies))

    def save(self):
        save_path = os.path.join(self.save_dir,
                                 str(self.i)+".pkl")
        th.save(self.reward_model.state_dict(), save_path)

    def _load_trajs(self, data_path):
        all_agents = os.listdir(data_path)
        assert len(self.use_data_from) > 0
        agent_types = ['ppo', 'sac', 'td3', 'ddpg']
        val_agents = []
        for at in agent_types:
            if at in self.use_data_from:
                val_agents += random.sample([x for x in all_agents if at.upper() in x],
                                                self.num_val_agents)
            else:
                val_agents += [x for x in all_agents if at.upper() in x]
        assert len(set(all_agents) - set(val_agents)) > 0
        self.train_trajs = []
        for agent_dir in os.listdir(data_path):
            if agent_dir == 'id.txt':
                continue
            trajs = []
            for rollout_dir in os.listdir(os.path.join(data_path, agent_dir)):
                for rollout_file in os.listdir(os.path.join(data_path, agent_dir, rollout_dir)):
                    path = os.path.join(data_path, agent_dir, rollout_dir, rollout_file)
                    if agent_dir not in val_agents:
                        self.train_trajs.append(utils.load_dict_from_pkl(path))
                    traj = flatten_trajectories(utils.load_dict_from_pkl(path))
                    traj['total_reward'] = np.sum(traj['rewards'])
                    trajs.append(traj)
            if agent_dir in val_agents:
                self.trajs['OOD_' + agent_dir] = deepcopy(trajs)
            else:
                self.trajs['IID_' + agent_dir] = deepcopy(trajs) 

    def _get_model_rewards(self, trajs):
        model_rewards = []
        for traj in trajs:
            rew  = self.reward_model(
                        state = traj['obs'],
                        action = traj['acs'],
                        next_state = traj['next_obs'],
                        done = traj['dones']).cpu().detach().numpy()
            model_rewards.append(np.sum(rew))
        return model_rewards


    def _compare_with_orcale_rank(self, plot=False):
        for data_type in self.trajs.keys():
            selected_trajs = self.trajs[data_type]
            oracle_rewards = [traj['total_reward'] for traj in selected_trajs]
            oracle_ranking = np.argsort(np.argsort(np.array(oracle_rewards)))
            model_rewards = np.array(self._get_model_rewards(selected_trajs))
            model_ranking = np.argsort(np.argsort(model_rewards))


            if plot:
                # Scatter plot of oracle_rewards vs model_rewards
                fig, ax = plt.subplots(figsize=(12,12))
                ax.scatter(oracle_rewards, model_rewards, s=400)
                ax.set_xlabel("Oracle")
                ax.set_ylabel("Reward Model")
                ax.set_title(str(self.i))
                wandb.log({(data_type[:3]+"_Rewards/"+data_type[3:]):wandb.Image(fig)}, step=self.i)
                plt.close(fig)

                # Scatter plot of oracle_ranking vs model_ranking
                fig, ax = plt.subplots(figsize=(12,12))
                ax.scatter(oracle_ranking, model_ranking, s=400)
                ax.set_xlabel("Oracle")
                ax.set_ylabel("Reward Model")
                ax.set_title(str(self.i))
                wandb.log({(data_type[:3]+"/"+data_type[3:]):wandb.Image(fig)}, step=self.i)
                plt.close(fig)

            hamming_dist = hamming(oracle_ranking, model_ranking)
            kendall_tau, _ = kendalltau(oracle_ranking, model_ranking)

            #self.logger.record(data_type+"/HammingDist", hamming_dist)
            #self.logger.record(data_type+"/KendallTau", kendall_tau)
            #self.logger.record(data_type+"/NumberOfTrajs", len(model_ranking))
            self.logger.record("KendallTau_"+data_type[:3]+"/"+data_type[3:], kendall_tau)


            # if the oracle ranking of trajectory is 'k' and the model ranking for
            # the same trajectory is in [k-m, k, k+m] we consider the error to be zero
            # otherwise we consider error to be 1
            # we do this for values of m = 2, 5, 10, 20, 30, 40
            #binwise_ranking_diff = dict(zip([2,5,10,20], [0,0,0,0]))
            #rank_diff = np.abs(oracle_ranking - model_ranking)
            #for k in binwise_ranking_diff.keys():
            #    assert k > 1 # implementation is not valid for k =1
            #    if k >= len(rank_diff):
            #        continue
            #    binwise_ranking_diff[k] = np.mean(rank_diff > k)
            #    self.logger.record(data_type+"/HammingDist_"+str(k), binwise_ranking_diff[k])


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


    eval_env = utils.make_eval_env(env_id=config.eval_env_id,
                                   normalize_obs=False,
                                   env_wrappers=env_wrappers,
                                   vec_wrappers=vec_wrappers)

    reward_model, reward_model_cb = setup_reward_model_and_cb(eval_env, config)

    # Logger
    sb_logger = logger.HumanOutputFormat(sys.stdout)

    rlhp = TrainRMfPrefsOffline(
            total_comparisons = config.total_comparisons,
            seed              = config.seed,
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
            save_all_reward_models = config.save_all_reward_models,
            disable_all_saving = config.disable_all_saving,
            save_every       = config.save_every,
            plot_every       = config.plot_every,
            eval_every       = config.eval_every,
            save_dir         = config.save_dir,
            data_dir         = config.data_dir,
            device           = config.device,
            use_data_from  = config.use_data_from,
            num_val_agents = config.num_val_agents,
                    )

    rlhp.train()

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
    parser.add_argument("--sync_wandb", "-sw", action="store_true",
        help="Sync at the end of experiment even if wandb offline.")
    # ======================== Environment ========================== #
    parser.add_argument("--train_env_id", "-tei", type=str, default="HalfCheetah-v3",
        help="Gym environment to train on.")
    parser.add_argument("--eval_env_id", "-eei", type=str, default="HalfCheetah-v3",
        help="Gym environment to eval on.")
    parser.add_argument("--seed", "-s", type=int, default=None)
    parser.add_argument("--rl_algo", "-rla", type=str, default="sac",
        choices=['ppo', 'sac', 'td3', 'ddpg'], help="Which RL algo to use?")
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
    parser.add_argument("--save_every", "-se", type=int, default=500)
    parser.add_argument("--plot_every", "-pe", type=int, default=500)
    parser.add_argument("--eval_every", "-ee", type=int, default=50)
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
    #================= Offline RLfHP ================================ #
    parser.add_argument('--use_data_from', type=str, default='pposactd3ddpg')
    parser.add_argument('--num_val_agents', type=int, default=2)
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
    # ======================= Expert Data =========================== #
    parser.add_argument('--data_dir', '-dd', type=str, default="data/HC",
        help="Path to expert or oracle data.")

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
