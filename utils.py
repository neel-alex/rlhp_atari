import torch as th
from torch import nn
from torch.utils.data.dataset import Dataset
from stungle_bungle3.common.env_util import make_vec_env
from stungle_bungle3.common.atari_wrappers import AtariWrapper
from stungle_bungle3.common.vec_env import DummyVecEnv, VecFrameStack
from stungle_bungle3.common.torch_layers import create_mlp
from stungle_bungle3.dqn.policies import QNetwork, CnnPolicy


def make_atari_env(env_id):
    env = make_vec_env(env_id, n_envs=1,
                       wrapper_class=AtariWrapper,
                       vec_env_cls=DummyVecEnv)
    env = VecFrameStack(env, n_stack=4)
    return env


class ExpertDataSet(Dataset):
    def __init__(self, states, actions, rewards, dones, next_steps=1):
        """
        Returns a state, action pair taken by the expert, as well as a list of
            rewards and subsequently encountered states of length next_steps,
            suitable for Q-learning purposes.
        Every value in dones corresponds to the index of a trajectory ending.
            If the requested list of rewards and next states goes past the end
            of a trajectory, the final valid state is repeated and extra
            rewards are set to 0.

        :param states: States encountered in order. len(states)=len(actions)+1
                        since there is a final state seen.
        :param actions: Actions taken by the agent in the corresponding state.
        :param rewards: Rewards corresponding to each state-action.
        :param dones: Time steps where a trajectory ended.
        :param next_steps: Number of subsequent steps to return, default 1.
        """
        self.states = th.from_numpy(states).float()
        self.state_shape = states.shape[1:]
        self.actions = th.from_numpy(actions).float()
        self.rewards = th.from_numpy(rewards).float()
        self.dones = dones
        self.next_steps = next_steps

    def __getitem__(self, index):
        rewards = th.empty((self.next_steps,), dtype=th.float32)
        next_states = th.empty((self.next_steps,) + self.state_shape, dtype=th.float32)
        # Track index, fill in space if we reach a done.
        j = 0
        for i in range(self.next_steps):
            if j + index in self.dones:
                rewards[i] = 0.
                next_states[i] = self.states[j + index]
                continue
            rewards[i] = self.rewards[i + index]
            next_states[i] = self.states[i + index + 1]
            j += 1

        return self.states[index], self.actions[index], rewards, next_states

    def __len__(self):
        return len(self.states) - 1


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
