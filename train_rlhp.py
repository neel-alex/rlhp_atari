from utils.env_utils import make_atari_env
from utils.dqn_utils import ExpertDQN, DuelingDQNPolicy, ExpertReplayBuffer
from utils.buffer_utils import BorjaCNN

import numpy as np
import torch as th

from preference_learning.rlfhp import TrainRM
from preference_learning.utils import TrajectoriesCollector

from stable_baselines3.common.vec_env import VecEnvWrapper, VecTransposeImage


class RewardModelWrapper(VecEnvWrapper):
    def __init__(self, venv, reward_model):
        super().__init__(venv)
        self.reward_model = reward_model
        self.last_obs = env.observation_space.sample()

    def step_wait(self):
        obs, rews, done, infos = self.venv.step_wait()
        rews = self.reward_model(th.tensor(self.last_obs, device='cuda').float()).cpu().detach().numpy()
        self.last_obs = obs
        return obs, rews, done, infos

    def reset(self):
        self.last_obs = self.venv.reset()
        return self.last_obs.copy()


env_id: str = "EnduroNoFrameskip-v4"
seed = 4
mini = True
buffer_size = 10000
discount = 0.99
learning_starts = 0
batch_size = 32
exploration_fraction = 0.1
device = 'cuda'
verbose = 1

env = make_atari_env(env_id)
eval_env = make_atari_env(env_id)
eval_env.seed(seed)

env = VecTransposeImage(env)
eval_env = VecTransposeImage(eval_env) 
reward_model = BorjaCNN(env.observation_space.shape, 1)
reward_model.to('cuda')

env = RewardModelWrapper(env, reward_model)

data = np.load(f"record/{env_id}_expert_data{'_mini' if mini else ''}.npz")
print(f"Loaded {len(data['actions'])} transitions.")


model = ExpertDQN(DuelingDQNPolicy,
                  env,
                  buffer_size=buffer_size,
                  gamma=discount,
                  learning_starts=learning_starts,
                  batch_size=batch_size,
                  replay_buffer_class=ExpertReplayBuffer,
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


trainer = TrainRM(seed=4,
                  total_comparisons=100,
                  init_pct_comparisons=0.5,
                  total_timesteps=100,
                  gathering_counts=10,
                  trajectories_collector=TrajectoriesCollector(model, eval_env),
                  fragment_length=25,
                  transition_oversampling=2,
                  temperature=50,
                  discount_factor=0.99,
                  return_prob=False,
                  model=reward_model,
                  noise_prob=0.0,
                  optimizer_type='Adam',
                  optimizer_kwargs=dict(lr=1e-3,
                                        betas=(0.9, 0.999),
                                        eps=1e-08,
                                        weight_decay=1e-5),
                  epochs_per_iter=20,
                  initial_epoch_multiplier=1,
                  batch_size=32,
                  save_dir=None,
                  save_every=1000,
                  device='cuda')

trainer.setup_training()

for i in range(100):
    model.learn(32)
    trainer.update(i)
    print(i)
