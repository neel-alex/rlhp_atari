from stable_baselines3.dqn import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from dqfp import eval_episode

env_id = "EnduroNoFrameskip-v4"
# env_id = "QbertNoFrameskip-v4"

# 1: Random Qbert, 2: Trained Qbert, 3: Random Enduro, 6: Trained Enduro (failed run)
run_number = 61
num_episodes = 10

model_path = f"results/dqfp/{run_number}/policy.zip"

env = make_vec_env(env_id,
                   n_envs=1,
                   wrapper_class=AtariWrapper,
                   vec_env_cls=DummyVecEnv)
env = VecFrameStack(env, n_stack=4)

custom_objects = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
}

model = DQN.load(model_path, env=env, custom_objects=custom_objects, device="cuda:0", seed=4)
deterministic = False

rewards = []

for i in range(num_episodes):
    print(f"Episode {i+1}...")
    reward = eval_episode(model, env)
    rewards.append(reward)

print(f"Average reward: {sum(rewards) / len(rewards)}")

