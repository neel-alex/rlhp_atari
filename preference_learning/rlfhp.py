import os
from tqdm import tqdm
import numpy as np
import torch as th
import math

from preference_learning.utils import (
                        FragmentsGenerator,
                        SyntheticGatherer,
                        PreferenceDataset,
                        flatten_trajectories,
                        preference_collate_fn)


class TrainRM:
    def __init__(
            self,
            seed,
            total_comparisons,
            init_pct_comparisons,
            total_timesteps,
            gathering_counts,
            trajectories_collector,
            fragment_length,
            transition_oversampling,
            # dummy human
            temperature,
            discount_factor,
            return_prob,
            # train
            model,
            noise_prob,
            optimizer_type,
            optimizer_kwargs,
            epochs_per_iter,
            initial_epoch_multiplier,
            batch_size,
            save_dir,
            save_every,
            device
    ):
        self.seed = seed
        self.total_comparisons = total_comparisons
        self.init_comparisons_pct = init_pct_comparisons
        self.noise_prob = noise_prob
        self.device = device

        self.total_timesteps = total_timesteps
        self.gathering_counts = gathering_counts

        self.tc = trajectories_collector

        # initialize objects to simulate preference collection
        # Fragments Generator
        self.frag_gen = FragmentsGenerator(fragment_length, seed)
        self.fragment_length = fragment_length
        self.transition_oversampling = transition_oversampling

        # Preference Collection
        self.discount_factor = discount_factor
        self.noise_prob = noise_prob
        self.dummy_human = SyntheticGatherer(
                        temperature=temperature,
                        discount_factor=discount_factor,
                        sample=not return_prob,
                        )

        # Making a dataset
        self.dataset = PreferenceDataset()

        # setup training
        self.reward_model = model
        self.batch_size = batch_size
        self.epochs = epochs_per_iter
        self.initial_epoch_multiplier = initial_epoch_multiplier
        if optimizer_type == 'Adam':
            self.optim = th.optim.Adam(
                             self.reward_model.parameters(),
                             **optimizer_kwargs)
        else:
            raise NotImplementedError

        self.save_every = save_every
        self.save_dir = None if save_dir is None else os.path.join(save_dir, 'reward_models')
        if self.save_dir:
            os.mkdir(self.save_dir)

        # Some variables for bookkeeping
        self.total_comparisons_so_far = 0
        self.total_rm_updates_so_far = 0
        self.updates = 0
        self.trajs = {'IID': []}
        self.last_time_trigger = 0
        self.train_every = -1
        self.comparisons_per_gather_iter = -1

    def setup_training(self):
        self.train_every = math.floor(self.total_timesteps / self.gathering_counts)
        comps_left_after_initial = self.total_comparisons * (1 - self.init_comparisons_pct)
        self.comparisons_per_gather_iter = int(comps_left_after_initial // self.gathering_counts)

    def update(self, rl_step):
        model_updated = 0
        if self.updates == 0:  # first time model being updated
            if self.init_comparisons_pct > 0.0:
                initial_comparisons = int(self.init_comparisons_pct *
                                          self.total_comparisons)
            else:
                initial_comparisons = self.comparisons_per_gather_iter
            self._collect_data(num_pairs=initial_comparisons)
            self.total_comparisons_so_far += initial_comparisons
            self._train(self.dataset, self.initial_epoch_multiplier)
            model_updated = 1
            self.last_time_trigger = rl_step

        if (rl_step - self.last_time_trigger) >= self.train_every:
            self._collect_data(num_pairs=self.comparisons_per_gather_iter)
            self.total_comparisons_so_far += self.comparisons_per_gather_iter
            self._train(self.dataset)
            model_updated = 1
            self.last_time_trigger = rl_step

        self.total_rm_updates_so_far += model_updated

        # saving
        if rl_step % self.save_every == 0:
            self.save()

        self.updates += 1

    def _collect_data(self, num_pairs):
        """
        A utility function which samples trajectories from the environment,
        fragments them, have dummy human annotate them and then adds them
        to the dataset.
        """
        steps = math.ceil(self.transition_oversampling * 2 *
                          num_pairs * self.fragment_length)
        trajs = self.tc.sample(steps)
        self.trajs['IID_Latest'] = []
        for traj in trajs:
            flat_traj = flatten_trajectories(traj)
            flat_traj['total_reward'] = np.sum(flat_traj['rewards'])
            self.trajs['IID'].append(flat_traj)
            self.trajs['IID_Latest'].append(flat_traj)
        fragments = self.frag_gen(trajs, num_pairs)
        preferences = self.dummy_human(fragments)
        self.dataset.push(fragments, preferences)
     
    def _loss(self, fragment_pairs: dict[str, list], preferences: np.ndarray) -> th.tensor:
        probs = th.empty(len(fragment_pairs), dtype=th.float32)
        for i, fragment in enumerate(fragment_pairs):
            frag1, frag2 = fragment
            trans1 = flatten_trajectories(frag1)
            trans2 = flatten_trajectories(frag2)
            rews1 = self.reward_model(th.tensor(trans1['obs'].squeeze(), device='cuda').float())
            rews2 = self.reward_model(th.tensor(trans2['obs'].squeeze(), device='cuda').float())
            probs[i] = self._probability(rews1, rews2)
        predictions: th.Tensor = (probs > 0.5).float()
        preferences_th: th.Tensor = th.as_tensor(preferences, dtype=th.float32)
        ground_truth = (preferences_th > 0.5).float()
        accuracy = (predictions == ground_truth).float().mean()
        return (th.nn.functional.binary_cross_entropy(probs, preferences_th),
                accuracy.item())

    def _probability(self, rews1: th.tensor, rews2: th.tensor, threshold=50) -> th.tensor:
        # assert rews1.ndim == rews2.ndim == 1
        if self.discount_factor == 1:
            returns_diff = (rews2 - rews1).sum()
        else: 
            discounts = self.discount_factor ** th.arange(len(rews1), device=self.device)
            returns_diff = (discounts * (rews2 - rews1)).sum()
        # clip to avoid overflows
        returns_diff = th.clip(returns_diff, -threshold, threshold)
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

    def save(self):
        if self.save_dir is not None:
            save_path = os.path.join(self.save_dir,
                                     str(self.total_rm_updates_so_far)+".pkl")
            th.save(self.reward_model.state_dict(), save_path)

    def _get_model_rewards(self, trajs):
        model_rewards = []
        for traj in trajs:
            rew = self.reward_model(
                        state=traj['obs'],
                        action=traj['acs'],
                        next_state=traj['next_obs'],
                        done=traj['dones']).cpu().detach().numpy()
            model_rewards.append(np.sum(rew))
        return model_rewards
