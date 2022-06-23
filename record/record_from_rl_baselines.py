import numpy as np
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

algo = "ppo"
# env_id = "BreakoutNoFrameskip-v4"
# env_id = "PongNoFrameskip-v4"
env_id = "EnduroNoFrameskip-v4"
# env_id = "QbertNoFrameskip-v4"
model_path = ("rl-baselines3-zoo/rl-trained-agents/"
              f"{algo}/{env_id}_1/{env_id}.zip"
              )

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

model_class = None
if algo == "a2c":
    model_class = A2C
elif algo == "ppo":
    model_class = PPO
elif algo == "dqn":
    model_class = DQN

model = model_class.load(model_path, env=env, custom_objects=custom_objects, device="cuda:0", seed=4)

obs = env.reset()
deterministic = False

num_frames = 40000

states = np.zeros((num_frames + 1,) + env.observation_space.shape)
actions = np.zeros((num_frames,) + env.action_space.shape)
rewards = np.zeros((num_frames,))
dones = []

for frame in range(num_frames):
    action, _ = model.predict(obs, deterministic=deterministic)
    states[frame] = obs
    actions[frame] = action
    obs, rew, done, info = env.step(action)
    rewards[frame] = rew
    # env.render("human")
    if done:
        dones.append(frame)
        # episode_infos = info[0].get("episode")
        # if episode_infos is not None:
        #     rewards.append(episode_infos['r'])
        obs = env.reset()
    print(frame, end="\r")

states[frame+1] = obs
dones.append(frame)
dones = np.array(dones)

# SB3 wants NCHW, not NHWC, easier to do it now
states = np.moveaxis(states, 3, 1)

print(states.shape,
      actions.shape,
      rewards.shape,
      dones.shape)

np.savez_compressed(
    f"{env_id}_expert_data",
    states=states,
    actions=actions,
    rewards=rewards,
    dones=dones,
)
