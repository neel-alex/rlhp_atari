import numpy as np
import gym
from stable_baselines3.dqn import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import cv2


env_id = "EnduroNoFrameskip-v4"
# env_id = "QbertNoFrameskip-v4"

# 1: Random Qbert, 2: Trained Qbert, 3: Random Enduro, 6: Trained Enduro
run_number = 4

model_path = f"results/dqfp/{run_number}/policy.zip"

frames = []


class FrameSaveWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        frames.append(observation)
        return observation


env = make_vec_env(env_id,
                   n_envs=1,
                   wrapper_class=lambda e: AtariWrapper(FrameSaveWrapper(e)),
                   vec_env_cls=DummyVecEnv)
env = VecFrameStack(env, n_stack=4)

custom_objects = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
}

model = DQN.load(model_path, env=env, custom_objects=custom_objects, device="cuda:0", seed=4)
deterministic = False

obs = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=deterministic)
    obs, rew, done, info = env.step(action)
    done = done[0]
    if done:
        if info[0]['lives'] != 0:
            obs = env.reset()
            done = False

filename = "record/video.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(filename, fourcc, 60, (frames[0].shape[1], frames[0].shape[0]))

for frame in frames:
    out.write(np.flip(frame, axis=-1))  # cv2 uses BGR
out.release()
