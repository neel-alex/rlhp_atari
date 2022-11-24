import numpy as np
import torch as th
from gym import spaces

from utils import make_atari_env

from dqn_utils import ExpertReplayBuffer, ExpertMarginDQN, DuelingDQNPolicy

from pl.rlhp.utils.reward_nets import setup_reward_model_and_cb

env_id = "EnduroNoFrameskip-v4"
env = make_atari_env(env_id)

from types import SimpleNamespace

def counter():
    a = 0
    while True:
        yield a
        a += 1

c = counter()

def count_reward(*args):
    num_rewards = args[0].shape[0]
    print(num_rewards, args[0].shape)
    return th.full((num_rewards,), next(c))

config = SimpleNamespace(
    reward_scheme='s',
    dont_use_reward_model=False,
    reward_net_type="mlp",
    device='cpu',
    reward_net_layers=[32, 32],
    reward_model_path=None,
    rl_algo="expert_dqn",
)

reward_model, _ = setup_reward_model_and_cb(env, config)

policy = ExpertMarginDQN(DuelingDQNPolicy,
                         env,
                         replay_buffer_kwargs={
                             "expert_observations": np.zeros((20,) + env.observation_space.shape),
                             "expert_actions": np.zeros((20,), env.action_space),
                             "expert_rewards": np.zeros((20, 1)),
                             "expert_next_observations": np.zeros((20,) + env.observation_space.shape),
                             "reward_model": reward_model,
                             "reward_relabeling": True,
                            },
                         learning_starts=100
                         )

policy.learn(20000)
