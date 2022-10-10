from gym.wrappers.pixel_observation import PixelObservationWrapper
import gym
import numpy as np

class PixelObservationWrapperCustom(PixelObservationWrapper):
    def __init__(self, env, pixels_only=True):
        super().__init__(env, pixels_only=pixels_only)
        if pixels_only:
            self.observation_space = self.observation_space['pixels']

    def observation(self, observation):
        return super().observation(observation)['pixels']


class RewardModelWrapper(gym.core.Wrapper):
    def __init__(self, env, reward_model):
        super().__init__(env)
        self.reward_model = reward_model

    def step(self, action):
        obs, rews, done, infos = self.env.step(action)

        rews = self.reward_model(self.last_obs[None, ...],
                                 action[None, ...],
                                 obs[None, ...],
                                 np.array(done)[None, ...]).cpu().detach().numpy()
        self.last_obs = obs
        return obs, rews, done, infos

    def reset(self):
        self.last_obs = self.env.reset()
        return self.last_obs.copy()
                                 

