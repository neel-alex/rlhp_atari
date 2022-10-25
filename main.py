import os
from typing import Type, Union

import numpy as np
from sacred import Experiment, observers
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy

from utils.env_utils import make_atari_env
from utils.dqn_utils import ExpertDQN, DuelingDQNPolicy
from utils.buffer_utils import BorjaReplayBuffer

edqn_experiment = Experiment("edqn")
observer = observers.FileStorageObserver('results/edqn')
edqn_experiment.observers.append(observer)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


@edqn_experiment.config
def config():
    mini: bool = False
    env_id: str = "EnduroNoFrameskip-v4"

    model_cls: Union[Type[DQN], str] = ExpertDQN
    policy_cls: Union[Type[DQNPolicy], str] = DuelingDQNPolicy

    discount: float = 0.99
    learning_starts: int = 0
    exploration_fraction: float = 0.1
    buffer_size: int = int(1e5)

    batch_size: int = 32
    scheduler_gamma: float = 1.0
    learning_rate: float = 6.25e-5
    eps: float = 1.5625e-4
    weight_decay: float = 0

    seed: int = 4  # chosen by fair dice roll. guaranteed to be random.

    verbose: int = 1
    device: str = "cuda"


@edqn_experiment.capture
def get_model_and_policy(model_cls, policy_cls):
    if type(model_cls) is str:
        if model_cls == "DQN":
            model_cls = DQN
        elif model_cls == "ExpertDQN":
            model_cls = ExpertDQN

    if type(policy_cls) is str:
        if policy_cls == "DQNPolicy":
            policy_cls = DQNPolicy
        elif policy_cls == "DuelingDQNPolicy":
            policy_cls = DuelingDQNPolicy

    return model_cls, policy_cls


@edqn_experiment.automain
def main(mini,
         env_id,
         discount,
         learning_starts,
         buffer_size,
         exploration_fraction,
         batch_size,
         device,
         seed,
         verbose):
    env = make_atari_env(env_id)
    eval_env = make_atari_env(env_id)
    eval_env.seed(seed)

    data = np.load(f"record/{env_id}_expert_data{'_mini' if mini else ''}.npz")
    print(f"Loaded {len(data['actions'])} transitions.")

    model_cls, policy_cls = get_model_and_policy()

    model = model_cls(policy_cls,
                      env,
                      buffer_size=buffer_size,
                      gamma=discount,
                      learning_starts=learning_starts,
                      batch_size=batch_size,
                      replay_buffer_class=BorjaReplayBuffer,
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

    model.learn(int(2e5))
    # model._setup_learn(1, None)
    # for i in range(500_001):
    #     model.train(1, 32)
    #     if i % 2500 == 0:
    #         polyak_update(model.q_net.parameters(), model.q_net_target.parameters(), model.tau)

    model.save(observer.dir + f"/final_model.ckpt")
