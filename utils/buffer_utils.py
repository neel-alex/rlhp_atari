from gym import spaces
import torch as th
from torch import nn
import numpy as np

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize

from typing import Union, NamedTuple, List, Dict, Any, Optional


class ExpertReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    n_step_observations: th.Tensor
    dones: th.Tensor
    expert_indices: th.Tensor  # True iff the corresponding sample is expert data, False otherwise.
    rewards: th.Tensor
    n_step_rewards: int


class ExpertReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        expert_observations: Optional[np.ndarray] = None,
        expert_actions: Optional[np.ndarray] = None,
        expert_rewards: Optional[np.ndarray] = None,
        expert_dones: Optional[np.ndarray] = None,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        n_forward: int = 3,
        optimize_memory_usage: bool = True,
        discount: float = 0.99,
    ):
        """ All expert_{} parameters need to have their length as the first dimension.
        Just append them to the start of observations?
        And reset the counter to the start of observations rather than the protected part of the dataset?
        """
        if expert_observations is None:
            self.num_expert = 0
        else:
            expert_observations = expert_observations[:-1]  # TODO: fix this hack.
            self.num_expert = expert_observations.shape[0]
        if not optimize_memory_usage:
            raise NotImplementedError("Sadly, you must optimize memory usage for this type of replay buffer.")
        super().__init__(buffer_size + self.num_expert,
                         observation_space,
                         action_space,
                         device=device,
                         n_envs=n_envs,
                         optimize_memory_usage=True,
                         handle_timeout_termination=False)
        # copy in expert data to the start of the buffer
        if expert_observations is not None:
            self.observations[:self.num_expert] = expert_observations.reshape((self.num_expert,
                                                                               *(self.observations.shape[1:])))
            self.actions[:self.num_expert] = expert_actions.reshape((self.num_expert,
                                                                    *(self.actions.shape[1:])))
            self.rewards[:self.num_expert] = expert_rewards.reshape((self.num_expert,
                                                                    *(self.rewards.shape[1:])))
            for done in expert_dones:
                self.dones[done] = 1

        self.dones[self.num_expert - 1] = 1  # TODO: is this fixable?

        self.pos = self.num_expert
        self.n_forward = n_forward
        self.discount = discount

    def reset(self) -> None:
        self.full = False
        self.pos = self.num_expert

    def add(self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]]) -> None:
        super().add(obs, next_obs, action, reward, done, infos)
        if self.full and self.pos == 0:
            # Buffer just looped over to 0, set pos to real start (ignoring expert actions)
            self.pos = self.num_expert

    def to_torch(self, array: Union[np.ndarray, th.Tensor], copy: bool = True) -> th.Tensor:
        if isinstance(array, th.Tensor):
            if copy:
                return array.detach().clone().to(self.device)
            else:
                return array.to(self.device)
        return super().to_torch(array, copy)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ExpertReplayBufferSamples:
        blacklist = {*np.where(self.dones)[0]}
        cap = max(0, self.pos-3)
        if self.full:
            blacklist.add(self.pos)
            cap = self.buffer_size - 1
        for b in list(blacklist):
            blacklist.add(b-1)
            blacklist.add(b-2)

        whitelist = [i for i in range(cap) if i not in blacklist]
        batch_inds = th.tensor(np.random.choice(whitelist, size=batch_size))
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: th.Tensor, env: Optional[VecNormalize] = None) -> ExpertReplayBufferSamples:
        next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
        n_step_obs = self._normalize_obs(self.observations[(batch_inds + 3) % self.buffer_size, 0, :], env)
        n_step_rewards = sum([self.discount ** i * self.rewards[(batch_inds + i) % self.buffer_size, 0]
                              for i in range(self.n_forward)])

        data = (
            self._normalize_obs(self.observations[batch_inds, 0, :], env),
            self.actions[batch_inds, 0, :],
            next_obs,
            n_step_obs,
            self.dones[batch_inds, 0],
            batch_inds < self.num_expert,
            self._normalize_reward(self.rewards[batch_inds, 0].reshape(-1, 1), env),
            self._normalize_reward(n_step_rewards)
        )
        r = tuple(map(self.to_torch, data))
        return ExpertReplayBufferSamples(*r)


class BorjaReplayBuffer(ExpertReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        expert_observations: Optional[np.ndarray] = None,
        expert_actions: Optional[np.ndarray] = None,
        expert_rewards: Optional[np.ndarray] = None,
        expert_dones: Optional[np.ndarray] = None,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        n_forward: int = 3,
        optimize_memory_usage: bool = True,
        discount: float = 0.99,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            expert_observations=expert_observations,
            expert_actions=expert_actions,
            expert_rewards=expert_rewards,
            expert_dones=expert_dones,
            device=device,
            n_envs=n_envs,
            n_forward=n_forward,
            optimize_memory_usage=optimize_memory_usage,
            discount=discount,
        )
        self.dummy_reward_net = BorjaCNN(self.observation_space.shape, 1)
        if th.device(self.device) == th.device('cuda'):
            self.dummy_reward_net.cuda()
            print("Creating reward net on GPU")

        SPLIT_SIZE = 200
        for i, observation_batch in enumerate(np.split(np.squeeze(self.observations), SPLIT_SIZE)):
            with th.no_grad():
                reward = self.dummy_reward_net(th.tensor(observation_batch).to(self.device).float() / 255.0)
                self.rewards[i*len(observation_batch):(i+1)*len(observation_batch),] = reward.to('cpu')

    def add(self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]]) -> None:
        s = reward.shape
        reward = self.dummy_reward_net(th.tensor(obs).to(self.device).float() / 255.0)
        reward = reward.to('cpu').numpy().reshape(s)
        super().add(obs, next_obs, action, reward, done, infos)


class BorjaCNN(nn.Module):
    # from https://github.com/uzman-anwar/preference-learning/blob/master/rlhp/utils/reward_nets.py
    def __init__(self, input_shape, features_dim):
        super().__init__()
        n_input_channels = input_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=7, stride=3, padding=0),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.2),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            x = th.rand(input_shape)
            n_flatten = self.cnn(th.as_tensor(x[None]).float()).shape[1]
        self.linear = nn.Linear(n_flatten, features_dim)

    def forward(self, x):
        return self.linear(self.cnn(x))
