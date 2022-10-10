"""Constructs deep network reward models.
   Highly "inspired" from https://github.com/HumanCompatibleAI/imitation/blob/master/src/imitation/rewards/reward_nets.py"""

import abc
from typing import Callable, Iterable, Sequence, Tuple, Optional

import gym
import numpy as np
import torch as th
from stable_baselines3.common import preprocessing
from stable_baselines3.common.torch_layers import create_mlp
from torch import nn

import sys
sys.path.append('/home/neel/projects/rlhp_atari/pl/')
# TODO(neel): horrendously hacky.... :(

from rlhp.rlfhp import RewardModelCallback


class RewardNet(nn.Module, abc.ABC):
    """Minimal abstract reward network.
    Only requires the implementation of a forward pass (calculating rewards given
    a batch of states, actions, next states and dones).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        normalize_images: bool = True,
    ):
        """Initialize the RewardNet.
        Args:
            observation_space: the observation space of the environment
            action_space: the action space of the environment
            normalize_images: whether to automatically normalize
                image observations to [0, 1] (from 0 to 255). Defaults to True.
        """
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.normalize_images = normalize_images

    @abc.abstractmethod
    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """Compute rewards for a batch of transitions and keep gradients."""

    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Preprocess a batch of input transitions and convert it to PyTorch tensors.
        The output of this function is suitable for its forward pass,
        so a typical usage would be ``model(*model.preprocess(transitions))``.
        Args:
            state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            action: The action input. Its shape is
                `(batch_size,) + action_space.shape`. The None dimension is
                expected to be the same as None dimension from `obs_input`.
            next_state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            done: Whether the episode has terminated. Its shape is `(batch_size,)`.
        Returns:
            Preprocessed transitions: a Tuple of tensors containing
            observations, actions, next observations and dones.
        """
        state_th = th.as_tensor(state, device=self.device)
        action_th = th.as_tensor(action, device=self.device)
        next_state_th = th.as_tensor(next_state, device=self.device)
        done_th = th.as_tensor(done, device=self.device)

        del state, action, next_state, done  # unused

        # preprocess
        state_th = preprocessing.preprocess_obs(
            state_th,
            self.observation_space,
            self.normalize_images,
        )
        action_th = preprocessing.preprocess_obs(
            action_th,
            self.action_space,
            self.normalize_images,
        )
        next_state_th = preprocessing.preprocess_obs(
            next_state_th,
            self.observation_space,
            self.normalize_images,
        )
        done_th = done_th.to(th.float32)

        n_gen = len(state_th)
        assert state_th.shape == next_state_th.shape
        assert len(action_th) == n_gen

        return state_th, action_th, next_state_th, done_th

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        """Compute rewards for a batch of transitions without gradients.
        Preprocesses the inputs, converting between Torch
        tensors and NumPy arrays as necessary.
        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.
        Returns:
            Computed rewards of shape `(batch_size,`).
        """
        # switch to eval mode (affecting normalization, dropout, etc)
        self.eval()

        state_th, action_th, next_state_th, done_th = self.preprocess(
            state,
            action,
            next_state,
            done,
        )

        rew_th = self.forward(state_th, action_th, next_state_th, done_th)
        rew = rew_th.detach().cpu().numpy().flatten()
        assert rew.shape == state.shape[:1]

        # switch back to train mode
        self.train()

        return rew

    @property
    def device(self) -> th.device:
        """Heuristic to determine which device this module is on."""
        try:
            first_param = next(self.parameters())
            return first_param.device
        except StopIteration:
            # if the model has no parameters, we use the CPU
            return th.device("cpu")

    @property
    def dtype(self) -> th.dtype:
        """Heuristic to determine dtype of module."""
        try:
            first_param = next(self.parameters())
            return first_param.dtype
        except StopIteration:
            # if the model has no parameters, default to float32
            return th.get_default_dtype()


class BasicRewardNet(RewardNet):
    """MLP that takes as input the state, action, next state and done flag.
    These inputs are flattened and then concatenated to one another. Each input
    can enabled or disabled by the `use_*` constructor keyword arguments.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        use_state: bool = True,
        use_action: bool = True,
        use_next_state: bool = False,
        use_done: bool = False,
        hidden_sizes: Optional[int] = [32, 32],
        device: str = "cpu",
        **kwargs,
    ):
        """Builds reward MLP.
        Args:
            observation_space: The observation space.
            action_space: The action space.
            use_state: should the current state be included as an input to the MLP?
            use_action: should the current action be included as an input to the MLP?
            use_next_state: should the next state be included as an input to the MLP?
            use_done: should the "done" flag be included as an input to the MLP?
            kwargs: passed straight through to `build_mlp`.
        """
        super().__init__(observation_space, action_space)
        combined_size = 0

        self.use_state = use_state
        if self.use_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_action = use_action
        if self.use_action:
            combined_size += preprocessing.get_flattened_obs_dim(action_space)

        self.use_next_state = use_next_state
        if self.use_next_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_done = use_done
        if self.use_done:
            combined_size += 1

        self.hidden_sizes = hidden_sizes

        self.mlp = nn.Sequential(
                *create_mlp(combined_size, 1, self.hidden_sizes),
        )
        self.mlp.to(device)

    def forward_offline(self, state, action, next_state, done):
        if len(state.shape) > 2:
            batch_size, frag_len, _ = state.shape
            state = th.reshape(state, [batch_size*frag_len, -1])
            action = th.reshape(action, [batch_size*frag_len, -1])
            next_state = th.reshape(next_state, [batch_size*frag_len, -1])
            done = th.reshape(done, [batch_size*frag_len, -1])

        inputs = []
        if self.use_state:
            inputs.append(state)
        if self.use_action:
            inputs.append(action)
        if self.use_next_state:
            inputs.append(next_state)
        if self.use_done:
            inputs.append(done)

        inputs_concat = th.cat(inputs, dim=1)
        
        outputs = self.mlp(inputs_concat)
        outputs = th.reshape(outputs, [batch_size, frag_len])

        return outputs


    def forward(self, state, action, next_state, done, use_preprocessing=True):
        if use_preprocessing:
            state, action, next_state, done = self.preprocess(
                    state, action, next_state, done)
        inputs = []

        if self.use_state:
            inputs.append(th.flatten(state, 1))
        if self.use_action:
            inputs.append(th.flatten(action, 1))
        if self.use_next_state:
            inputs.append(th.flatten(next_state, 1))
        if self.use_done:
            inputs.append(th.reshape(done, [-1, 1]))

        inputs_concat = th.cat(inputs, dim=1)
        outputs = self.mlp(inputs_concat).squeeze()

        return outputs

class BorjaCNN(nn.Module):
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

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, x):
        # adds gaussian noise
        with th.no_grad():
            x += th.normal(0, 0.1, size=x.shape, device=x.device)
        return self.linear(self.cnn(x))

class NatureCNN(nn.Module):
    def __init__(self, input_shape, features_dim):
        super().__init__()
        n_input_channels = input_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            x = th.rand(input_shape)
            n_flatten = self.cnn(th.as_tensor(x[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, x):
        return self.linear(self.cnn(x))


class BasicCNNRewardNet(RewardNet):
    """CNN that takes as input the state, action, next state and done flag.
    These inputs are flattened and then concatenated to one another. Each input
    can enabled or disabled by the `use_*` constructor keyword arguments.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        use_state: bool = True,
        use_action: bool = True,
        use_next_state: bool = False,
        use_done: bool = False,
        cnn_type: bool = 'borja_cnn',
        features_dim: int = 64,
        hidden_sizes: Optional[int] = [64],
        device: str = "cpu",
        **kwargs,
    ):
        """
        Builds reward NN. It projects state observation and next state obsevation
        (which are assumed to be images) NatureCNN to embedding of size x
        through separate NatureCNN networks.
        The image embeddings are then concatenated with action
        and done and passed through a small MLP (whose size you can control through
        hidden_sizes parameter.
        Args:
            observation_space: The observation space.
            action_space: The action space.
            use_state: should the current state be included as an input to the MLP?
            use_action: should the current action be included as an input to the MLP?
            use_next_state: should the next state be included as an input to the MLP?
            use_done: should the "done" flag be included as an input to the MLP?
            kwargs: passed straight through to `build_mlp`.
        """
        super().__init__(observation_space, action_space)
        combined_size = 0

        self.use_state = use_state
        if self.use_state:
            state_shape = observation_space.sample().shape
            assert len(state_shape) > 2, \
                    "If using CCN reward net, use pixel based observations"
            if cnn_type == 'nature_cnn':
                self.state_cnn = NatureCNN(state_shape,
                                           features_dim)
            elif cnn_type == 'borja_cnn':
                self.state_cnn = BorjaCNN(state_shape, features_dim)
            else:
                raise NotImplementedError
            self.state_cnn.to(device)
            combined_size += features_dim

        self.use_next_state = use_next_state
        if self.use_next_state:
            state_shape = observation_space.sample().shape
            assert len(state_shape) > 2, \
                    "If using CCN reward net, use pixel based observations"
            if cnn_type == 'nature_cnn':
                self.next_state_cnn = NatureCNN(state_shape,
                                                features_dim)
            elif cnn_type == 'borja_cnn':
                self.state_cnn = BorjaCNN(state_shape,
                                           features_dim)
            else:
                raise NotImplementedError
            self.next_state_cnn.to(device)
            combined_size += features_dim

        self.use_action = use_action
        if self.use_action:
            combined_size += preprocessing.get_flattened_obs_dim(action_space)

        self.use_done = use_done
        if self.use_done:
            combined_size += 1

        self.hidden_sizes = hidden_sizes

        self.mlp = nn.Sequential(
                *create_mlp(combined_size, 1, self.hidden_sizes),
        )
        self.mlp.to(device)

    def forward(self, state, action, next_state, done, use_preprocessing=True):
        if use_preprocessing:
            state, action, next_state, done = self.preprocess(
                    state, action, next_state, done)
        inputs = []
        if self.use_state:
            inputs.append(th.flatten(self.state_cnn(state), 1))
        if self.use_action:
            inputs.append(th.flatten(action, 1))
        if self.use_next_state:
            inputs.append(th.flatten(self.next_state_cnn(next_state), 1))
        if self.use_done:
            inputs.append(th.reshape(done, [-1, 1]))

        inputs_concat = th.cat(inputs, dim=1)

        outputs = self.mlp(inputs_concat).squeeze()

        return outputs

def setup_reward_model_and_cb(train_env, config):
    # Initialize reward model
    if config.reward_scheme == 's':
        use_state, use_action, use_next_state, use_done = True, False, False, False
    elif config.reward_scheme == 'sa':
        use_state, use_action, use_next_state, use_done = True, True, False, False
    elif config.reward_scheme == 'ss':
        use_state, use_action, use_next_state, use_done = True, False, True, False
    elif config.reward_scheme == 'sas':
        use_state, use_action, use_next_state, use_done = True, True, True, False
    elif config.reward_scheme == 'sasd':
        use_state, use_action, use_next_state, use_done = True, True, True, True
    else:
        raise NotImplementedError

    if config.dont_use_reward_model:
        return None, None

    if config.reward_net_type == 'mlp':
        reward_model = BasicRewardNet(train_env.observation_space,
                                      train_env.action_space,
                                      use_state=use_state,
                                      use_action=use_action,
                                      use_next_state=use_next_state,
                                      use_done=use_done,
                                      device=config.device,
                                      mlp_hidden_sizes=config.reward_net_layers)
    else:
        reward_model = BasicCNNRewardNet(train_env.observation_space,
                                         train_env.action_space,
                                         use_state=use_state,
                                         use_action=use_action,
                                         use_next_state=use_next_state,
                                         use_done=use_done,
                                         cnn_type=config.reward_net_type,
                                         features_dim=config.reward_net_conv_features_dim,
                                         device=config.device,
                                         mlp_hidden_sizes=config.reward_net_layers)

    if config.reward_model_path is not None:
        if not config.override_dont_train_saved_rm_assertion:
            assert config.dont_train_reward_model
        reward_model.load_state_dict(
            th.load(config.reward_model_path))
        # TODO: add set_device method which ensures that model gets
        #       transferred to the desired device correctly
        reward_model.to(config.device)

    buffer_type = 'rollout' if config.rl_algo in ['ppo', 'trpo'] else 'replay'

    if config.rl_algo in ['ppo', 'trpo']:
        reward_model_cb = RewardModelCallback(
                reward_model = reward_model,
                normalize_reward = (not config.dont_normalize_reward),
                buffer_type  = buffer_type)
    else:
        reward_model_cb = None

    return reward_model, reward_model_cb

