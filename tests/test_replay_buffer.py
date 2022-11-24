import numpy as np
import torch as th
from gym import spaces

from dqn_utils import ExpertReplayBuffer


def counter():
    a = 0
    while True:
        yield a
        a += 1


c = counter()


def count_reward(*args):
    num_rewards = args[0].shape[0]
    return th.full((num_rewards, 1), next(c))


buffer = ExpertReplayBuffer(reward_model=count_reward,
                            reward_relabeling=True,
                            buffer_size=10,
                            observation_space=spaces.Box(low=0., high=1., shape=(2, 2)),
                            action_space=spaces.Discrete(10),
                            expert_observations=np.ones((4, 2, 2)),
                            expert_actions=np.ones((4, 1)),
                            expert_rewards=np.ones((4, 1)),
                            expert_next_observations=np.zeros((4, 2, 2))
                            )

buffer.add(obs=np.full((2, 2), 2, dtype=np.int),
           next_obs=np.ones((2, 2), dtype=np.int),
           action=np.ones((1,), dtype=np.int),
           reward=np.ones((1,), dtype=np.int),
           done=np.zeros((1,), dtype=np.int))

buffer.sample(5)
