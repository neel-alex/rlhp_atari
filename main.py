# Test re-upload

import numpy as np
import torch as th
from torch.utils.data.dataset import random_split
from torch.utils.data.dataloader import DataLoader
from stable_baselines3.dqn import DQN
from sacred import Experiment, observers

from dqfp import train_policy, test_policy
from utils import make_atari_env, ExpertDataSet

dqfp_experiment = Experiment("dqfp")
observer = observers.FileStorageObserver('results/dqfp')
dqfp_experiment.observers.append(observer)


@dqfp_experiment.config
def config():
    env_id = "EnduroNoFrameskip-v4"
    device = "cuda"
    dataloader_kwargs = {"num_workers": 1, "pin_memory": True}

    scheduler_gamma = 1.0
    learning_rate = 1e-3
    epochs = 20
    batch_size = 32
    th.manual_seed(4)


@dqfp_experiment.automain
def main(env_id, device, dataloader_kwargs, scheduler_gamma, learning_rate, epochs, batch_size):
    env = make_atari_env(env_id)
    model = DQN("CnnPolicy", env, verbose=1)
    data = np.load(f"record/{env_id}_expert_data.npz")
    dataset = ExpertDataSet(data['states'],
                            data['actions'],
                            data['rewards'],
                            data['dones'],
                            next_steps=3)

    train_size = int(0.8 * len(dataset))
    split = train_size, len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, split)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, shuffle=True, **dataloader_kwargs)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size, shuffle=True, **dataloader_kwargs)

    test_policy(model, test_loader, device=device)
    model.policy = train_policy(model, train_loader, epochs, device=device,
                                learning_rate=learning_rate, scheduler_gamma=scheduler_gamma)
    test_policy(model, test_loader, device=device)
    print(observer.dir)

    model.save(observer.dir + "/policy")

