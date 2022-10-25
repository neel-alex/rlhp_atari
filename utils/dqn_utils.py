from typing import Union, Type, Dict, Tuple, Any, Optional

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import QNetwork, CnnPolicy, DQNPolicy
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.buffers import ReplayBuffer

from utils.buffer_utils import ExpertReplayBuffer


def large_margin_loss(action_qs, target_action, expert_inds=None, margin=0.8, device='auto'):
    # Mask out non-expert actions...
    if expert_inds is None:
        # Use all of them by default
        expert_inds = th.ones(target_action.shape[0], dtype=th.bool).to(device)

    # Get Qs of expert actions
    expert_margin = th.full(action_qs.shape, margin).to(device)
    margins = th.zeros(target_action.shape).to(device)
    expert_margin = expert_margin.scatter(1, target_action, margins)
    margin_adjusted_qs = action_qs + expert_margin
    best_qs, _ = th.max(margin_adjusted_qs, 1)

    expert_qs = th.gather(action_qs, -1, target_action)
    expert_qs = expert_qs.reshape(best_qs.shape)
    masked_diffs = expert_inds * (best_qs - expert_qs)
    return th.mean(masked_diffs)


def calculate_loss(replay_data, n_forward, gamma, q_net_target, q_net, device):
    with th.no_grad():
        # Compute the next Q-values using the target network
        next_q_values = q_net_target(replay_data.next_observations)
        next_q_values, _ = next_q_values.max(dim=1)
        next_q_values = next_q_values.reshape(-1, 1)
        if n_forward > 1:
            n_step_q_values = q_net_target(replay_data.n_step_observations)
            n_step_q_values, _ = n_step_q_values.max(dim=1)
            n_step_q_values = n_step_q_values.reshape(-1, 1)

    # Get current Q-values estimates
    all_q_values = q_net(replay_data.observations)
    current_q_values = th.gather(all_q_values, dim=1, index=replay_data.actions.long())

    # Compute the expert margin loss if applicable
    loss = large_margin_loss(all_q_values, replay_data.actions,
                             expert_inds=replay_data.expert_indices, device=device)

    # 1-step TD target
    target_q_values = replay_data.rewards + gamma * next_q_values
    # Compute Huber loss (less sensitive to outliers)
    loss += F.smooth_l1_loss(current_q_values, target_q_values)

    if n_forward > 1:
        # n-step TD target
        n_step_target_q_values = replay_data.n_step_rewards.reshape(-1, 1) + \
                                 gamma ** n_forward * n_step_q_values
        loss += F.smooth_l1_loss(current_q_values, n_step_target_q_values)

    return loss


class ExpertDQN(DQN):
    # Use margin loss + n_step q loss...
    # ExpertReplayBuffer
    # And reward model! See Usman's other code for implementation...
    #   Actually, don't think I need to? Reward model in replay buffer handles this.
    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = Type[ExpertReplayBuffer],
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = True,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        if replay_buffer_kwargs:
            replay_buffer_kwargs["discount"] = gamma
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,  # 1e6
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model
        )
        self.replay_buffer = self.replay_buffer  # type: Optional[ExpertReplayBuffer]
        self.extras = {}  # TODO: remove all extras if unnecessary?

        self.n_calls = 0
        self.train_losses = []

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            loss = calculate_loss(replay_data, self.replay_buffer.n_forward, self.gamma,
                                  self.q_net_target, self.q_net, self.device)

            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


class DuelingQNetwork(QNetwork):
    def __init__(self, observation_space, action_space, features_extractor, features_dim,
                 adv_net_arch=None, val_net_arch=None, activation_fn=nn.ReLU, normalize_images=True):
        super().__init__(
            observation_space,
            action_space,
            features_extractor,
            features_dim,
            net_arch=None,
            activation_fn=activation_fn,
            normalize_images=normalize_images
        )

        if adv_net_arch is None:
            adv_net_arch = [64, 64]
        if val_net_arch is None:
            val_net_arch = [64, 64]

        self.adv_net_arch, self.val_net_arch = adv_net_arch, val_net_arch
        action_dim = self.action_space.n  # number of actions
        adv_layers = create_mlp(self.features_dim, action_dim, self.adv_net_arch, self.activation_fn)
        val_layers = create_mlp(self.features_dim, 1, self.val_net_arch, self.activation_fn)
        adv_net, val_net = nn.Sequential(*adv_layers), nn.Sequential(*val_layers)
        self.q_net = DuelingQNetworkHelper(adv_net, val_net, action_dim)


class DuelingQNetworkHelper(nn.Module):
    def __init__(self, adv_net, val_net, action_dim):
        super().__init__()
        self.adv_net = adv_net
        self.val_net = val_net
        self.action_dim = action_dim

    def forward(self, x):
        adv = self.adv_net(x)
        val = self.val_net(x)

        return adv + (val - adv.mean(1, keepdim=True)).expand(-1, self.action_dim)


# Building off CnnPolicy for convenience...
# TODO: More modular so it plays nice with SB3 policies?
class DuelingDQNPolicy(CnnPolicy):
    def __init__(self, observation_space, action_space, lr_schedule,
                 adv_net_arch=None, val_net_arch=None, **kwargs):
        self.adv_net_arch = adv_net_arch
        self.val_net_arch = val_net_arch
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )

    def get_net_args(self):
        return {
                "observation_space": self.observation_space,
                "action_space": self.action_space,
                "adv_net_arch": self.adv_net_arch,
                "val_net_arch": self.val_net_arch,
                "activation_fn": self.activation_fn,
                "normalize_images": self.normalize_images,
                }

    def make_q_net(self):
        net_args = self.get_net_args()
        net_args = self._update_features_extractor(net_args, features_extractor=None)
        return DuelingQNetwork(**net_args)
