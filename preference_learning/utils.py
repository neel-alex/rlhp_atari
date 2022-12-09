from tqdm import tqdm
import numpy as np
import torch as th
import random
import pickle
from typing import Optional, Tuple

from scipy import special


def flatten_trajectories(traj):
    flat_traj = {}
    for k in traj.keys():
        if k == 'terminal':
            continue
        flat_traj[k] = np.concatenate(traj[k], axis=0)
    return flat_traj


class TrajectoriesCollector:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def sample(self, steps):
        obs = self.env.reset()
        trajs, i, i_episode = [], 0, 0
        traj = {"obs": [], "next_obs": [], "acs": [], "rewards": [], "dones": [],
                "terminal": False}
        with tqdm(total=steps) as pbar:
            while True:
                action = self.agent.predict(obs)[0]
                next_obs, reward, done, infos = self.env.step(action)
                traj["obs"].append(obs[None, ...])  # add batch dim
                traj["next_obs"].append(next_obs[None, ...])
                traj["acs"].append(action[None, ...])
                traj["rewards"].append(np.array((reward,)))
                traj["dones"].append(np.array((done,)))
                i += 1
                i_episode += 1
                obs = next_obs
                if done:
                    pbar.update(i_episode)
                    i_episode = 0
                    traj["terminal"] = True
                    trajs.append(traj)
                    traj = {"obs": [], "next_obs": [], "acs": [], "rewards": [],
                            "dones": [], "terminal": False}
                    obs = self.env.reset()
                    if i > steps:
                        break
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
            end = start + self.fragment_length
            dones = np.zeros((1, self.fragment_length))
            dones[-1] = 1 if ((end == n) and traj['terminal']) else 0
            fragment = {'obs':      traj['obs'][start:end],
                        'next_obs': traj['next_obs'][start:end],
                        'acs':      traj['acs'][start:end],
                        'rewards':  traj['rewards'][start:end],
                        'dones':    dones}
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
        # print(len(fragment_pairs[0][0]['obs']))
        rews1, rews2 = zip(
            *[
                (
                    discounted_sum(np.array(f1['rewards']), self.discount_factor),
                    discounted_sum(np.array(f2['rewards']), self.discount_factor),
                )
                for f1, f2 in fragment_pairs
            ],
        )
        return (np.squeeze(np.array(rews1, dtype=np.float32)),
                np.squeeze(np.array(rews2, dtype=np.float32)))


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


# RL - taken from imitation
def discounted_sum(arr, gamma):
    """Calculate the discounted sum of `arr`.
    If `arr` is an array of rewards, then this computes the return;
    however, it can also be used to e.g. compute discounted state
    occupancy measures.
    Args:
        arr: 1 or 2-dimensional array to compute discounted sum over.
            Last axis is timestep, from current time step (first) to
            last timestep (last). First axis (if present) is batch
            dimension.
        gamma: the discount factor used.
    Returns:
        The discounted sum over the timestep axis. The first timestep is undiscounted,
        i.e. we start at gamma^0.
    """
    # We want to calculate sum_{t = 0}^T gamma^t r_t, which can be
    # interpreted as the polynomial sum_{t = 0}^T r_t x^t
    # evaluated at x=gamma.
    # Compared to first computing all the powers of gamma, then
    # multiplying with the `arr` values and then summing, this method
    # should require fewer computations and potentially be more
    # numerically stable.
    arr = arr.squeeze()
    assert arr.ndim in (1, 2)
    if gamma == 1.0:
        return arr.sum(axis=0)
    else:
        return np.polynomial.polynomial.polyval(gamma, arr)
