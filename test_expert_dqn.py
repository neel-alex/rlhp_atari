import os

import numpy as np
from sacred import Experiment, observers
from stable_baselines3.dqn.policies import DQNPolicy

from utils import make_atari_env
from dqn_utils import ExpertMarginDQN, DuelingDQNPolicy

edqn_experiment = Experiment("edqn")
observer = observers.FileStorageObserver('results/edqn')
edqn_experiment.observers.append(observer)

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


@edqn_experiment.config
def config():
    mini = False
    env_id = "EnduroNoFrameskip-v4"
    discount = 0.99
    learning_starts = 100
    exploration_fraction = 0.1

    batch_size = 32

    seed = 4  # chosen by fair dice roll. guaranteed to be random.

    verbose = 1
    device = "cuda"


def eval_model(model, env, num_episodes=5, experiment=None, step=None, verbose=True):
    rewards = []
    for i in range(num_episodes):
        rewards.append(eval_episode(model, env))
    avg_reward = sum(rewards) / len(rewards)
    if experiment is not None:
        experiment.log_scalar("test.return", avg_reward, step=step)
    if verbose:
        print(f"Average Reward ({step if step is not None else ''}): {avg_reward:.2f}")


def eval_episode(model, env):
    reward = 0
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(action)
        done = done[0]
        if done:
            episode_infos = info[0].get("episode")
            if episode_infos is not None:
                reward += episode_infos['r']
            if info[0]['lives'] != 0:
                obs = env.reset()
                done = False
    return reward


def make_logger(env):
    def log(model, train_losses, n_batches, log_interval):
        eval_model(model, env, experiment=edqn_experiment, step=n_batches)
        edqn_experiment.log_scalar('train.loss', sum(train_losses)/len(train_losses), step=n_batches)
        if n_batches % (log_interval * 10) == 0:
            model.policy.save(observer.dir + f"/{n_batches}_batches_policy.ckpt")

    return log


@edqn_experiment.automain
def main(mini, env_id, discount, learning_starts, exploration_fraction, batch_size, device, seed, verbose):
    env = make_atari_env(env_id)
    eval_env = make_atari_env(env_id)
    log_function = make_logger(eval_env)

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
                            log_function=log_function,
                            )

    model.learn(2_000_500)
    model.policy.save(observer.dir + f"/final_policy.ckpt")
    print("Test")
