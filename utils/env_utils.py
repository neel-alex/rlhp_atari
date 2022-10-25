from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


def make_atari_env(env_id):
    env = make_vec_env(env_id, n_envs=1,
                       wrapper_class=AtariWrapper,
                       vec_env_cls=DummyVecEnv)
    env = VecFrameStack(env, n_stack=4)
    return env


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


def eval_model(model, env, num_episodes=5, experiment=None, step=None, verbose=True):
    rewards = []
    for i in range(num_episodes):
        rewards.append(eval_episode(model, env))
    avg_reward = sum(rewards) / len(rewards)
    if experiment is not None:
        experiment.log_scalar("test.return", avg_reward, step=step)
    if verbose:
        print(f"Average Reward ({step if step is not None else ''}): {avg_reward:.2f}")