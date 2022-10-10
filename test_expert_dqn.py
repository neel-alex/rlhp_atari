import os
import numpy as np
import torch as th
from gym import spaces
from sacred import Experiment, observers

from utils import make_atari_env
from dqn_utils import ExpertMarginDQN, DuelingDQNPolicy

edqn_experiment = Experiment("edqn")
observer = observers.FileStorageObserver('results/edqn')
edqn_experiment.observers.append(observer)

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


@edqn_experiment.config
def config():
    mini = True
    env_id = "EnduroNoFrameskip-v4"
    discount = 0.99
    learning_starts = 100
    exploration_fraction = 0.1

    batch_size = 32

    seed = 4  # chosen by fair dice roll. guaranteed to be random.

    verbose = 1
    device = "cuda"


@edqn_experiment.automain
def main(mini, env_id, discount, learning_starts, exploration_fraction, batch_size, device, seed, verbose):
    env = make_atari_env(env_id)

    data = np.load(f"record/{env_id}_expert_data{'_mini' if mini else ''}.npz")
    print(f"Loaded {len(data['actions'])} transitions.")

    model = ExpertMarginDQN(DuelingDQNPolicy,
                            env,
                            buffer_size=10_000,  # 2x mini
                            gamma=discount,
                            learning_starts=learning_starts,
                            batch_size=batch_size,
                            replay_buffer_kwargs={
                                "expert_observations": data['states'],
                                "expert_actions": data['actions'],
                                "expert_rewards": data['rewards'],
                                "expert_dones": data['dones'],
                                "n_forward": 3,
                               },
                            exploration_fraction=exploration_fraction,
                            device=device,
                            optimize_memory_usage=True,
                            seed=seed,
                            verbose=verbose,
                            )

    model.learn(100000)
    print("Test")
