import torch as th
from torch.utils.data.dataset import Dataset
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


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


def make_atari_env(env_id):
    env = make_vec_env(env_id, n_envs=1,
                       wrapper_class=AtariWrapper,
                       vec_env_cls=DummyVecEnv)
    env = VecFrameStack(env, n_stack=4)
    return env
