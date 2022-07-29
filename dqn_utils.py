from torch import nn
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.dqn.policies import QNetwork, CnnPolicy
from stable_baselines3 import DQN
from pl.stable_baselines3.common.buffers import ReplayBufferWithRM


class ExpertMarginDQN(DQN):
    # Use margin loss + n_step q loss...
    # ExpertReplayBuffer
    # And reward model! See Usman's other code for implementation...
    pass


class ExpertReplayBuffer(ReplayBufferWithRM):
    # Get obs, next obs, 3x next obs, expert action...
    # What else?
    pass


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
