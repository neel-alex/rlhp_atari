import os

import numpy as np
from sacred import Experiment, observers

from utils.env_utils import make_atari_env
from utils.dqn_utils import ExpertMarginDQN, DuelingDQNPolicy, BorjaReplayBuffer

edqn_experiment = Experiment("edqn")
observer = observers.FileStorageObserver('results/edqn')
edqn_experiment.observers.append(observer)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


@edqn_experiment.config
def config():
    mini = False
    env_id = "EnduroNoFrameskip-v4"
    discount = 0.99
    learning_starts = 0
    exploration_fraction = 0.1
    buffer_size = 0

    batch_size = 32
    scheduler_gamma = 1.0
    learning_rate = 6.25e-5
    eps = 1.5625e-4
    weight_decay = 0

    seed = 4  # chosen by fair dice roll. guaranteed to be random.

    verbose = 1
    device = "cuda"


@edqn_experiment.automain
def main(mini, env_id, discount, learning_starts, buffer_size, exploration_fraction, batch_size, device, seed, verbose):
    env = make_atari_env(env_id)
    eval_env = make_atari_env(env_id)
    eval_env.seed(seed)

    data = np.load(f"record/{env_id}_expert_data{'_mini' if mini else ''}.npz")
    print(f"Loaded {len(data['actions'])} transitions.")

    model = ExpertMarginDQN(DuelingDQNPolicy,
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
                            log_function=log_function,
                            )

    # model.learn(2e6)
    # model._setup_learn(1, None)
    # for i in range(500_001):
    #     model.train(1, 32)
    #     if i % 2500 == 0:
    #         polyak_update(model.q_net.parameters(), model.q_net_target.parameters(), model.tau)

    model.save(observer.dir + f"/final_model.ckpt")
    print("Test")
