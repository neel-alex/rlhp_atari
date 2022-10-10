from __future__ import annotations  # for backward compatibility
import os, time, wandb
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import random, math
from typing import Optional, Callable, Tuple
from functools import partial

from stable_baselines3.common import callbacks
from stable_baselines3.common.running_mean_std import RunningMeanStd

import sys
sys.path.append('/home/neel/projects/rlhp_atari/pl/rlhp')
# TODO(neel): horrendously hacky.... :(

import general_utils as utils
from scipy import special
from scipy.stats import kendalltau
from scipy.spatial.distance import hamming

## TODO: split the code across fies
## TODO: should we collect new data or use data from buffers for feedback?
## TODO: PreferenceDataset should have separate train and validation/test sets

def flatten_trajectories(traj):
    flat_traj = {}
    for k in traj.keys():
        if k == 'terminal':
            continue
        flat_traj[k] = np.concatenate(traj[k], axis=0)
    return flat_traj


class TrajectoriesCollector:
    def __init__(self, make_env_fn, parallel=False, allow_variable_horizon=False,
                 device='cpu'):
        if parallel or (not allow_variable_horizon):
            raise NotImplementedError
        self.make_env_fn = make_env_fn
        self.env = make_env_fn()
        self.device = device

    def sample(self, policy, steps):
        obs = self.env.reset()
        trajs, i, i_episode = [], 0, 0
        traj = {"obs":[], "next_obs": [], "acs": [], "rewards": [], "dones": [],
                "terminal":False}
        policy_device = policy.device
        if not th.device(self.device) == policy_device:
            policy = policy.to(th.device(self.device))
        with tqdm(total=steps) as pbar:
            while True:
                action = policy._predict(
                            (th.from_numpy(obs).to(self.device)),
                            deterministic=False).detach().cpu().numpy()
                next_obs, reward, done, infos = self.env.step(action)
                traj["obs"].append(obs)
                traj["next_obs"].append(next_obs)
                traj["acs"].append(action)
                traj["rewards"].append(np.array(reward))
                traj["dones"].append(np.array(done))
                i += 1
                i_episode += 1
                obs = next_obs
                if done:
                    pbar.update(i_episode)
                    i_episode = 0
                    traj["terminal"] = True
                    trajs.append(traj)
                    traj = {"obs":[], "next_obs": [], "acs": [], "rewards": [],
                            "dones":[], "terminal":False}
                    obs = self.env.reset()
                    if i > steps:
                        break
        if not th.device(self.device) == policy_device:
            policy = policy.to(policy_device)
        return trajs

class FragmentsGenerator:
    def __init__(self, fragment_length, seed):
        self.fragment_length = fragment_length
        self.rng = random.Random(seed)
        pass

    def _validate_trajs(self, trajs):
        for traj in trajs:
            assert len(traj['obs']) >= self.fragment_length

    def __call__(self, trajs, num_pairs):
        """
        Args:
            trajs: list of trajectories.
        """
        self._validate_trajs(trajs)
        weights = [len(traj) for traj in trajs]
        fragments = []
        for _ in range(2*num_pairs):
            # first sample a trajectory
            traj = self.rng.choices(trajs, weights, k=1)[0]
            n = len(traj['obs'])
            # sample start position
            start = self.rng.randint(0, n - self.fragment_length)
            end   = start + self.fragment_length
            dones = np.zeros((1, self.fragment_length))
            dones[-1] = 1 if ((end == n) and traj['terminal']) else 0
            fragment = dict(
                    obs = traj['obs'][start : end],
                    next_obs = traj['next_obs'][start : end],
                    acs = traj['acs'][start : end],
                    rewards = traj['rewards'][start : end],
                    dones = dones
                        )
            fragments.append(fragment)
        # fragments is currently a list of single fragments. We want to pair up
        # fragments to get a list of (fragment1, fragment2) tuples. To do so,
        # we create a single iterator of the list and zip it with itself:
        iterator = iter(fragments)
        return list(zip(iterator, iterator))


class SyntheticGatherer:
    """Computes synthetic preferences using ground-truth environment rewards."""

    def __init__(
        self,
        temperature: float = 1,
        discount_factor: float = 1,
        sample: bool = True,
        seed: Optional[int] = None,
        threshold: float = 50,
        custom_logger: Optional = None,
    ):
        """Initialize the synthetic preference gatherer.
        Args:
            temperature: the preferences are sampled from a softmax, this is
                the temperature used for sampling. temperature=0 leads to deterministic
                results (for equal rewards, 0.5 will be returned).
            discount_factor: discount factor that is used to compute
                how good a fragment is. Default is to use undiscounted
                sums of rewards (as in the DRLHP paper).
            sample: if True (default), the preferences are 0 or 1, sampled from
                a Bernoulli distribution (or 0.5 in the case of ties with zero
                temperature). If False, then the underlying Bernoulli probabilities
                are returned instead.
            seed: seed for the internal RNG (only used if temperature > 0 and sample)
            threshold: preferences are sampled from a softmax of returns.
                To avoid overflows, we clip differences in returns that are
                above this threshold (after multiplying with temperature).
                This threshold is therefore in logspace. The default value
                of 50 means that probabilities below 2e-22 are rounded up to 2e-22.
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        self.temperature = temperature
        self.discount_factor = discount_factor
        self.sample = sample
        self.rng = np.random.default_rng(seed=seed)
        self.threshold = threshold
        self.logger = custom_logger

    def __call__(self, fragment_pairs):
        """Computes probability fragment 1 is preferred over fragment 2."""
        returns1, returns2 = self._reward_sums(fragment_pairs)
        if self.temperature == 0:
            return (np.sign(returns1 - returns2) + 1) / 2

        returns1 /= self.temperature
        returns2 /= self.temperature

        # clip the returns to avoid overflows in the softmax below
        returns_diff = np.clip(returns2 - returns1, -self.threshold, self.threshold)
        # Instead of computing exp(rews1) / (exp(rews1) + exp(rews2)) directly,
        # we divide enumerator and denominator by exp(rews1) to prevent overflows:
        model_probs = 1 / (1 + np.exp(returns_diff))
        # Compute the mean binary entropy. This metric helps estimate
        # how good we can expect the performance of the learned reward
        # model to be at predicting preferences.
        entropy = -(
            special.xlogy(model_probs, model_probs)
            + special.xlogy(1 - model_probs, 1 - model_probs)
        ).mean()
        if self.logger is not None:
            self.logger.record("RM Training/entropy", entropy)

        if self.sample:
            return self.rng.binomial(n=1, p=model_probs).astype(np.float32)
        else:
            return model_probs

    def _reward_sums(self, fragment_pairs) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: Should this be fragment_length + 1 ?
        #print(len(fragment_pairs[0][0]['obs']))
        rews1, rews2 = zip(
            *[
                (
                    utils.discounted_sum(np.array(f1['rewards']), self.discount_factor),
                    utils.discounted_sum(np.array(f2['rewards']), self.discount_factor),
                )
                for f1, f2 in fragment_pairs
            ],
        )
        return np.squeeze(np.array(rews1, dtype=np.float32)), \
                np.squeeze(np.array(rews2, dtype=np.float32))



class PreferenceDataset(th.utils.data.Dataset):
    def __init__(self):
        self.fragments1 = []
        self.fragments2 = []
        self.preferences = np.array([])

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

    def __getitem__(self, i):
        return (self.fragments1[i], self.fragments2[i]), self.preferences[i]

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


def preference_collate_fn(batch: tuple[dict[str, list], np.ndarray]):
    fragment_pairs, preferences = zip(*batch)
    return list(fragment_pairs), np.array(preferences)

class RewardModelCallback(callbacks.BaseCallback):
    """
    This callback overwrites rewards in rollout_buffers (ppo etc)
    by those obtained from the reward_model callable given in the
    input.

    Using this callback instead of wrapper over gym environment is
    much faster as calls to NN based reward_model can be batched
    and processed on GPU quickly.
    """
    def __init__(
        self,
        reward_model: Callable,
        normalize_reward: bool,
        buffer_type: str,
        verbose: int = 1
        ):
        """
        Args:
        reward_model: A callable that returns the reward for a transition
            when queried.
        normalize_reward: If True, rewards returned from reward model
            are normalized before being written to buffer.
        verbose: [sb callback parameter to control log level? #TODO
        """
        super().__init__(verbose)
        self.reward_model = reward_model

        # Setup reward normalization
        self.normalize_reward = normalize_reward
        self.ret_rms = RunningMeanStd(shape=())
        self.ret = None

        self.buffer_type = buffer_type

        self.last_buffer_pos = 0

    def _init_callback(self):
        # Buffer
        if self.buffer_type == 'replay':
            self.buffer = self.model.replay_buffer
        elif self.buffer_type == 'rollout':
            self.buffer = self.model.rollout_buffer

    def _on_step(self):
        pass

    def _normalize_rewards(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        # TODO: is this function ok for SAC?
        buffer_size, n_envs = rewards.shape
        normalized_reward = np.zeros((buffer_size, n_envs))
        if self.ret is None or self.buffer_type in ['ppo', 'trpo']:
            self.ret = np.zeros(n_envs)
        for i in range(buffer_size):
            r = rewards[i,:]
            self.ret = self.ret*self.training_env.gamma + r
            self.ret_rms.update(self.ret)
            # TODO: Should we clip reward too?
            normalized_reward[i,:] = r/np.sqrt(self.ret_rms.var + 1e-4)
            self.ret[dones[i,:]] = 0
        return normalized_reward

    def _on_rollout_end(self):
        if self.buffer_type == 'rollout':
            idxs = list(range(0, self.buffer.buffer_size))
            self._relabel_data(idxs)
        else:
            raise NotImplementedError

    def _relabel_data(self, idxs):
        """
        From the buffer take the samples specified by idxs and
        relabel their rewards using the reward model.
        """
        obs = self._drop_envs_dim(self.buffer.observations[idxs].copy())
        acs = self._drop_envs_dim(self.buffer.actions[idxs].copy())
        next_obs = self._drop_envs_dim(self.buffer.next_observations[idxs].copy())
        dones = self._drop_envs_dim(self.buffer.dones[idxs].copy())

        # unnormalize obs only for PPO
        # SAC already stores unnormalized obs
        if self.buffer_type == 'rollout':
            obs = self.training_env.unnormalize_obs(obs)
            next_obs = self.training_env.unnormalize_obs(next_obs)

        # get reward model rewards
        new_rewards = self.reward_model(obs, acs, next_obs,
                                        dones).detach().cpu().numpy()
        new_rewards = self._add_envs_dim(new_rewards)[:,:,0]
        dones = self._add_envs_dim(dones)[:,:,0]
        assert new_rewards.shape == self.buffer.rewards[idxs].shape

        # should log un-normalized reward
        self.logger.record("Reward Model/Mean Reward", float(np.mean(new_rewards)))
        self.logger.record("Reward Model/Max Reward", float(np.max(new_rewards)))
        self.logger.record("Reward Model/Min Reward", float(np.min(new_rewards)))
        if self.normalize_reward:
            new_rewards = self._normalize_rewards(new_rewards, dones.astype(bool))

        self.buffer.rewards[idxs] = new_rewards

        # recompute returns and advantages in case of ppo/trpo
        if self.buffer_type == 'rollout':
            last_values, dones = (self.model.extras['last_values'],
                                  self.model.extras['dones'])
            self.buffer.compute_returns_and_advantage(last_values, dones)

        del obs, acs, next_obs, dones, new_rewards

    def _drop_envs_dim(self, x: np.ndarray) -> np.ndarray:
        """
        For arbitrary inputs of shape [buffer_size, n_envs, ...],
        it drops n_envs dimension and returns [buffer_size*n_envs, -1].
        """
        x_shape = x.shape
        self._latest_buffer_size = x_shape[0]
        self._latest_n_envs = x_shape[1]
        new_x_shape = (x_shape[0]*x_shape[1],) + x.shape[2:]
        return np.reshape(x, new_x_shape)

    def _add_envs_dim(self, x: np.ndarray) -> np.ndarray:
        """
        For arbitrary inputs of shape [batch_size*n_envs, ...],
        it [batch_size, n_envs, -1].
        """
        return np.reshape(x, (self._latest_buffer_size, self._latest_n_envs, -1))



class TrainRMfPrefsCallback(callbacks.BaseCallback):
    """
    This callback trains a reward model from binary preferences collected
    from a (dummy) human on trajectories collected from  an agent acting
    in the environment.
    """
    def __init__(
        self,
        seed: int,
        rl_algo: int,
        # data collection parameters
        total_comparisons: int,
        total_timesteps: int,
        init_comparisons_pct: float,
        use_demos : bool,
        gathering_counts: int, # how many times collect trajs from env
                              # and get human to label them
                              # I deviate here from imitation
        transition_oversampling: int,
        ### trajectories collector
        make_sampling_env_fn: Callable,
        parallel_data_collection: bool,
        allow_variable_horizon: bool,
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
        anneal_old_data: bool,
        save_all_reward_models: bool,
        disable_all_saving: bool,
        save_dir: str,
        data_dir: str,
        initial_epoch_multiplier: int = 20,
        threshold: int = 50,
        verbose: int = 1,
        device: str = 'cpu'):
        """
        Args:
        ============  data collection parameters ================
        total_comparisons: Number of total comparisons to perform.
        total_timesteps: Number of steps RL agent will take in the env.
        init_comparisons_pct: Percentage of total comparisons that
            should be used to kickstart the reward model training;
            these comparisons are collected on trajectories produced
            by a randomly initialized RL agent.
        use_demos : Whether demonstrations are provided or not?
            In case demonstrations are provided, they may be used
            in two ways:
            (1) Use behaviour cloning to initialize RL policy
            (2) To add artificial comparisons to dataset by making
                comparisons between trajectories produced by a random
                RL agent and these demos and prefering these demos.
            Right now; code does not support functionality to use demos.
        transition_oversampling: this is the factor by which we oversample
            transitions/steps.
        gathering_counts: This parameter controls how many times (and how often)
            agent is rolled out in the sampling environment to produce trajectories;
            collect preferences on produced trajectories and then update the reward
            model.
            Specifically, our trajectories collection strategy is as follows:
                - if initi_comprision_pct is > 0. we gather data that
                  is sufficient for init_comparisons_pct*total_comparisons
                  and train reward model on it.
                - Rest of comparisons budget is equally divided into
                  gathering_counts.
            This strategy is different from imitation where the relevant input
            parameter is comparisons_per_iter. That strategy can not be used
            here due to differences in code design.
        =============== TrajectoriesCollector args ===============

        make_sampling_env_fn: Function to make sampling env in which RL
            agent acts to produce trajectores.
        parallel_data_collection: Should we use parallel processing
            to collect trajectories?
        allow_variable_horizon: If False (default), algorithm will raise an
            exception if it detects trajectories of different length during
            training. If True, overrides this safety check. WARNING: variable
            horizon episodes leak information about the reward via termination
            condition, and can seriously confound evaluation. Read
            https://imitation.readthedocs.io/en/latest/guide/variable_horizon.html
            before overriding this.

        ============== FragmentGenerator args ===================
        fragment_length: How long should the fragments be?

        ============== SyntheticGatherer args ===================
        temperature: the preferences are sampled from a softmax, this is
            the temperature used for sampling. temperature=0 leads to deterministic
            results (for equal rewards, 0.5 will be returned).
        discount_factor: discount factor that is used to compute
            how good a fragment is.
        return_prob: if False, the preferences are 0 or 1, sampled from
            a Bernoulli distribution (or 0.5 in the case of ties with zero
            temperature). If True, then the underlying Bernoulli probabilities
            are returned instead.
        threshold: preferences are sampled from a softmax of returns.
            To avoid overflows, we clip differences in returns that are
            above this threshold (after multiplying with temperature).
            This threshold is therefore in logspace. The default value
            of 50 means that probabilities below 2e-22 are rounded up

        ============= Training arguments ========================
        model: Reward Model (should be nn.Module) which should be updated/
            trained.
        noise_prob: assumed probability with which the preference
            is uniformly random (used for the model of preference generation
            that is used for the loss)
            # Usman's note: I don't totally understand how this impacts
            # reward model training
        weight_decay: weight_decay strength for optimizer of reward model.
        optimizer_type: which optimizer class to use to optimze reward model.
        epochs: how many epochs a time to train reward model. Note that
            cumulatively reward model will be trained on
            (gathering_counts + 1*initial_epoch_multiplier) * epochs
            where +1 goes away if init_comparisons_pct = 0.0
        initial_epoch_multiplier: the first time reward model is trained, we
            train it for initial_epoch_multiplier*epochs
        batch_size: number of fragment pairs per batch
        learning_rate: Learning rate
        anneal_old_data: If True, an "age" parameter is associated with
            each sample in PreferenceDataset and samples with older age
            are sampled less often.
        """
        super().__init__(verbose)

        self.rl_algo = rl_algo
        self.total_comparisons = total_comparisons
        self.total_timesteps   = total_timesteps
        assert init_comparisons_pct > 0.0
        self.init_comparisons_pct = init_comparisons_pct
        self.use_demos         = use_demos
        self.discount_factor = discount_factor
        self.threshold = threshold
        self.temperature = temperature
        self.noise_prob = noise_prob
        self.gathering_counts = gathering_counts
        self.device = device

        if self.use_demos:
            raise NotImplementedError

        # initialize objects to simulate preference collection
        ## Trajectory Collector
        self.tc = TrajectoriesCollector(
                        make_env_fn=make_sampling_env_fn,
                        parallel=parallel_data_collection,
                        allow_variable_horizon=allow_variable_horizon,
                        device='cpu')

        ## Fragments Generator
        self.frag_gen = FragmentsGenerator(fragment_length, seed)
        self.fragment_length = fragment_length
        self.transition_oversampling = transition_oversampling

        ## Preference Collection
        self.dummy_human = SyntheticGatherer(
                        temperature=temperature,
                        discount_factor=discount_factor,
                        sample=not return_prob,
                        custom_logger=self.logger)

        ## Making a dataset
        self.dataset = PreferenceDataset()

        # setup training
        self.reward_model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_epoch_multiplier = initial_epoch_multiplier
        self.anneal_old_data = anneal_old_data
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
        self.trajs['IID'] = []
        if data_dir is not None:
            self._load_trajs(data_dir)

        # Some variables for bookkeeping
        self.i = 0 # counter to check which rl_iter we are on
        self.total_comparisons_so_far = 0
        self.total_rm_updates_so_far = 0

    def _collect_data(self, num_pairs: int =50):
        """
        A utility function which samples trajectories from the environment,
        fragments them, have dummy human annotate them and then adds them
        to the dataset.
        """
        policy = self.model.policy #partial(self.model.policy._predict, deterministic=False)
        steps = math.ceil(
                self.transition_oversampling * 2 * num_pairs * \
                    self.fragment_length)
        self.logger.log(f"Collecting {steps} trajectory steps")
        trajs = self.tc.sample(policy, steps)
        self.trajs['IID_Latest'] = []
        for traj in trajs:
            flat_traj = flatten_trajectories(traj)
            flat_traj['total_reward'] = np.sum(flat_traj['rewards'])
            self.trajs['IID'].append(flat_traj)
            self.trajs['IID_Latest'].append(flat_traj)
        fragments = self.frag_gen(trajs, num_pairs)
        preferences = self.dummy_human(fragments)
        if self.anneal_old_data:
            raise NotImplementedError
        self.dataset.push(fragments, preferences)
        


    def _on_step(self):
        pass

    def _init_callback(self):
        pass

    def _setup_training(self):
        """
        Uses gathering_counts and other input parameters to determine
        on which callback calls should collection+training step
        be executed.
        """
        # Get number of timesteps RL agent will do in env at one time
        # This is equal to capacity of rollout buffer
        if self.rl_algo in ['ppo', 'trpo']:
            rl_steps = np.prod(self.model.rollout_buffer.observations.shape[:2])
            rl_iters = math.ceil(self.total_timesteps / rl_steps)
        elif self.rl_algo in ['sac', 'td3', 'ddpg', 'edqn']:
            frequency = self.model.train_freq if type(self.model.train_freq) == int else self.model.train_freq.frequency
            rl_iters = (self.total_timesteps - self.model.learning_starts)/frequency
        # We can not do gathering_counts equal steps of RM training
        # unless the callback is called at least or more than
        # gathering_counts times
        assert rl_iters >= self.gathering_counts, \
                "Insufficent timesteps."
        # We distribute gathering_counts as follow
        # We look for greatest perfect divisor of rl_iters which is
        # smalles than gathering_counts
        # This gives us 'update_every' parameter
        # Rest of the gathering_counts are distributed randomly
        # across rest of the rl_iters
        self.original_gathering_counts = self.gathering_counts
        if self.init_comparisons_pct > 0.0:
            self.gathering_counts -= 1
        while (not rl_iters % self.gathering_counts == 0):
            self.gathering_counts -= 1
        self.additional_training_iters = []
        diff = self.original_gathering_counts - self.gathering_counts
        while diff > 0:
            k = np.random.randint(rl_iters)
            if k % self.gathering_counts == 0 and k != 0:
                self.additional_training_iters.append(k)
                diff -= 1
        self.update_every = int(rl_iters/self.gathering_counts)
        self.comparisons_per_gather_iter = \
                int((self.total_comparisons * (1 - self.init_comparisons_pct))
                    // self.original_gathering_counts)

    def _on_rollout_end(self):
        model_updated = 0
        self._compare_with_orcale_rank()
        if self.i == 0:
            self._setup_training()
            if self.init_comparisons_pct > 0.0:
                initial_comparisons = int(self.init_comparisons_pct * \
                                            self.total_comparisons)
            else:
                initial_comparisons = self.comparisons_per_gather_iter
            self._collect_data(num_pairs=initial_comparisons)
            self.total_comparisons_so_far += initial_comparisons
            self._train(self.dataset, self.initial_epoch_multiplier)
            self._compare_with_orcale_rank()
            model_updated = 1

        elif self.i % self.update_every == 0 or self.i in self.additional_training_iters:
            self._collect_data(num_pairs=self.comparisons_per_gather_iter)
            self.total_comparisons_so_far += self.comparisons_per_gather_iter
            self._train(self.dataset)
            self._compare_with_orcale_rank()
            model_updated = 1

        self.total_rm_updates_so_far += model_updated

        # saving
        if not self.disable_all_saving and \
            (self.save_all_reward_models or
             self.total_rm_updates_so_far == self.gathering_counts):
            self.save()


        self.logger.record("RM Training/Iteration", self.i)
        self.logger.record("RM Training/Model Updated", model_updated)
        self.logger.record("RM Training/Total Model Updates",
                                self.total_rm_updates_so_far)
        self.logger.record("RM Training/Total Comparisons",
                                self.total_comparisons_so_far)
        self.i += 1

    def _loss(self, fragment_pairs: dict[str, list], preferences: np.ndarray) -> th.tensor:
        probs = th.empty(len(fragment_pairs), dtype=th.float32)
        for i, fragment in enumerate(fragment_pairs):
            frag1, frag2 = fragment
            trans1 = flatten_trajectories(frag1)
            trans2 = flatten_trajectories(frag2)
            rews1  = self.reward_model(
                        state = trans1['obs'],
                        action = trans1['acs'],
                        next_state = trans1['next_obs'],
                        done = trans1['dones'])
            rews2  = self.reward_model(
                        state = trans2['obs'],
                        action = trans2['acs'],
                        next_state = trans2['next_obs'],
                        done = trans2['dones'])
            probs[i] = self._probability(rews1, rews2)
        predictions = (probs > 0.5).float()
        preferences_th = th.as_tensor(preferences, dtype=th.float32)
        ground_truth = (preferences_th > 0.5).float()
        accuracy = (predictions == ground_truth).float().mean()
        return (th.nn.functional.binary_cross_entropy(probs, preferences_th),
                accuracy.item())

    def _probability(self, rews1: th.tensor, rews2: th.tensor) -> th.tensor:
        assert rews1.ndim == rews2.ndim == 1
        if self.discount_factor == 1:
            returns_diff = (rews2 - rews1).sum()
        else: 
            discounts = self.discount_factor ** th.arange(len(rews1), device=self.device)
            returns_diff = (discounts * (rews2 - rews1)).sum()
        # clip to avoid overflows
        returns_diff = th.clip(returns_diff, -self.threshold, self.threshold)
        model_probability = (1 / (1 + returns_diff.exp()))
        return self.noise_prob * 0.5 + (1 - self.noise_prob)*model_probability

    def _train(self, dataset: PreferenceDataset, epoch_multiplier: float = 1.0):
        """Trains for `epoch_multiplier * self.epochs` epochs over `dataset`."""
        dataloader = th.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=preference_collate_fn,
        )
        epochs = round(self.epochs * epoch_multiplier)
        for _ in tqdm(range(epochs)):
            losses, accuracies = [], []
            for fragment_pairs, preferences in dataloader:
                self.optim.zero_grad()
                loss, accuracy = self._loss(fragment_pairs, preferences)
                loss.backward()
                self.optim.step()
                losses.append(loss.item())
                accuracies.append(accuracy)
            self.logger.record("RM Training/Loss", np.mean(losses))
            self.logger.record("RM Training/Accuracy", np.mean(accuracies))

        # For off-policy-algos like SAC, we need to update the rewards
        # for all entries in replay buffer once we update the RM.
        # The only way we can communicate between calbacks is through
        # the main class, so, we will use a custom defined dict in the
        # main class for this.
        self.model.extras['relabel_complete_buffer'] = True

    def save(self):
        save_path = os.path.join(self.save_dir,
                                 str(self.total_rm_updates_so_far)+".pkl")
        th.save(self.reward_model.state_dict(), save_path)

    def _load_trajs(self, data_path):
        all_agents = os.listdir(data_path)
        for agent_dir in os.listdir(data_path):
            if agent_dir == 'id.txt':
                continue
            self.trajs['OOD_' + agent_dir] = []
            for rollout_dir in os.listdir(os.path.join(data_path, agent_dir)):
                for rollout_file in os.listdir(os.path.join(data_path, agent_dir, rollout_dir)):
                    path = os.path.join(data_path, agent_dir, rollout_dir, rollout_file)
                    traj = flatten_trajectories(utils.load_dict_from_pkl(path))
                    #traj = (utils.load_dict_from_pkl(path))
                    traj['total_reward'] = np.sum(traj['rewards'])
                    self.trajs['OOD_' + agent_dir].append(traj)
        #self.table = wandb.Table(columns=list(range(len(self.trajs))))
        #self.table.add_data(*deepcopy(self.oracle_ranking))

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


    def _compare_with_orcale_rank(self):
        return None # TODO: figure out this crash...: /home/neel/projects/rlhp_atari/pl/wandb/ABC/wandb/offline-run-20220824_150528-3t1c17t0/files/output.log
        for data_type in self.trajs.keys():
            oracle_rewards = [traj['total_reward'] for traj in self.trajs[data_type]]
            oracle_ranking = np.argsort(np.argsort(oracle_rewards))
            model_rewards = np.array(self._get_model_rewards(self.trajs[data_type]))
            model_ranking = np.argsort(np.argsort(model_rewards))

            # Scatter plot of oracle_rewards vs model_rewards
            fig, ax = plt.subplots(figsize=(12,12))
            ax.scatter(oracle_rewards, model_rewards, s=400)
            ax.set_xlabel("Oracle")
            ax.set_ylabel("Reward Model")
            ax.set_title(str(self.i))
            wandb.log({data_type[:3]+"_Rewards/"+data_type[3:]:wandb.Image(fig)}, step=self.model.num_timesteps)
            plt.close(fig)

            # Scatter plot of oracle_ranking vs model_ranking
            fig, ax = plt.subplots(figsize=(12,12))
            ax.scatter(oracle_ranking, model_ranking, s=400)
            ax.set_xlabel("Oracle")
            ax.set_ylabel("Reward Model")
            ax.set_title(str(self.i))
            wandb.log({data_type[:3]+"/"+data_type[3:]:wandb.Image(fig)}, step=self.model.num_timesteps)
            plt.close(fig)

            #hamming_dist = hamming(oracle_ranking, model_ranking)
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



