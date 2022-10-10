import os, pickle
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable

import gym
import numpy as np
import torch as th

#from stable_baselines3.common import base_class, logger  # pytype: disable=pyi-error
from stable_baselines3.common import logger  # pytype: disable=pyi-error
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import (DummyVecEnv, VecEnv,
                                              sync_envs_normalization)


class BaseCallback(ABC):
    """
    Base class for callback.

    :param verbose:
    """

    def __init__(self, verbose: int = 0):
        super(BaseCallback, self).__init__()
        # The RL model
        self.model = None  # type: Optional[base_class.BaseAlgorithm]
        # An alias for self.model.get_env(), the environment used for training
        self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        self.n_calls = 0  # type: int
        # n_envs * n times env.step() was called
        self.num_timesteps = 0  # type: int
        self.verbose = verbose
        self.locals: Dict[str, Any] = {}
        self.globals: Dict[str, Any] = {}
        self.logger = None
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        self.parent = None  # type: Optional[BaseCallback]

    # Type hint as string to avoid circular import
    def init_callback(self, model: "base_class.BaseAlgorithm") -> None:
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        self.model = model
        self.training_env = model.get_env()
        self.logger = logger
        self._init_callback()

    def _init_callback(self) -> None:
        pass

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        # Those are reference and will be updated automatically
        self.locals = locals_
        self.globals = globals_
        self._on_training_start()

    def _on_training_start(self) -> None:
        pass

    def on_rollout_start(self) -> None:
        self._on_rollout_start()

    def _on_rollout_start(self) -> None:
        pass

    @abstractmethod
    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """
        return True

    def on_step(self) -> bool:
        """
        This method will be called by the model after each call to ``env.step()``.

        For child callback (of an ``EventCallback``), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        self.n_calls += 1
        # timesteps start at zero
        self.num_timesteps = self.model.num_timesteps

        return self._on_step()

    def on_training_end(self) -> None:
        self._on_training_end()

    def _on_training_end(self) -> None:
        pass

    def on_rollout_end(self) -> None:
        self._on_rollout_end()

    def _on_rollout_end(self) -> None:
        pass

    def update_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        self.locals.update(locals_)
        self.update_child_locals(locals_)

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables on sub callbacks.

        :param locals_: the local variables during rollout collection
        """
        pass


class EventCallback(BaseCallback):
    """
    Base class for triggering callback on event.

    :param callback: Callback that will be called
        when an event is triggered.
    :param verbose:
    """

    def __init__(self, callback: Optional[BaseCallback] = None, verbose: int = 0):
        super(EventCallback, self).__init__(verbose=verbose)
        self.callback = callback
        # Give access to the parent
        if callback is not None:
            self.callback.parent = self

    def init_callback(self, model: "base_class.BaseAlgorithm") -> None:
        super(EventCallback, self).init_callback(model)
        if self.callback is not None:
            self.callback.init_callback(self.model)

    def _on_training_start(self) -> None:
        if self.callback is not None:
            self.callback.on_training_start(self.locals, self.globals)

    def _on_event(self) -> bool:
        if self.callback is not None:
            return self.callback.on_step()
        return True

    def _on_step(self) -> bool:
        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback is not None:
            self.callback.update_locals(locals_)


class CallbackList(BaseCallback):
    """
    Class for chaining callbacks.

    :param callbacks: A list of callbacks that will be called
        sequentially.
    """

    def __init__(self, callbacks: List[BaseCallback]):
        super(CallbackList, self).__init__()
        assert isinstance(callbacks, list)
        self.callbacks = callbacks

    def _init_callback(self) -> None:
        for callback in self.callbacks:
            callback.init_callback(self.model)

    def _on_training_start(self) -> None:
        for callback in self.callbacks:
            callback.on_training_start(self.locals, self.globals)

    def _on_rollout_start(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_start()

    def _on_step(self) -> bool:
        continue_training = True
        for callback in self.callbacks:
            # Return False (stop training) if at least one callback returns False
            continue_training = callback.on_step() and continue_training
        return continue_training

    def _on_rollout_end(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_end()

    def _on_training_end(self) -> None:
        for callback in self.callbacks:
            callback.on_training_end()

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        for callback in self.callbacks:
            callback.update_locals(locals_)


class CheckpointCallback(EventCallback):
    """
    Callback for saving a model every ``save_freq`` steps

    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param callback_on_new_save: callback to trigger when model is saved
    """

    def __init__(
            self,
            save_freq: int,
            save_path: str,
            name_prefix="rl_model",
            verbose=0,
            callback_on_new_save: Optional[BaseCallback] = None
        ):
        super(CheckpointCallback, self).__init__(callback_on_new_save, verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
            if self.callback is not None:
                self._on_event()
        return True


class ConvertCallback(BaseCallback):
    """
    Convert functional callback (old-style) to object.

    :param callback:
    :param verbose:
    """

    def __init__(self, callback, verbose=0):
        super(ConvertCallback, self).__init__(verbose)
        self.callback = callback

    def _on_step(self) -> bool:
        if self.callback is not None:
            return self.callback(self.locals, self.globals)
        return True


class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every eval_freq call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param deterministic: Whether to render or not the environment during evaluation
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str = None,
        best_model_save_path: str = None,
        deterministic: bool = True,
        render: bool = False,
        label_str: str = 'eval',
        x_str: str = '',
        verbose: int = 1,
        callback_for_evaluate_policy: Optional[Callable] = None
    ):
        super(EvalCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.label_str = label_str
        if self.label_str != 'eval':
            assert x_str != ''
        self.x_str = x_str
        self.callback_for_evaluate_policy = callback_for_evaluate_policy

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        if isinstance(eval_env, VecEnv):
            assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []

    def _init_callback(self):
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                callback=self.callback_for_evaluate_policy,
                deterministic=self.deterministic,
                return_episode_rewards=True
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record(self.label_str+"/mean_reward"+self.x_str, float(mean_reward))
            self.logger.record(self.label_str+"/mean_ep_length"+self.x_str, mean_ep_length)
            self.logger.record(self.label_str+"/best_mean_reward"+self.x_str, max(self.best_mean_reward, mean_reward))

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


class StopTrainingOnRewardThreshold(BaseCallback):
    """
    Stop the training once a threshold in episodic reward
    has been reached (i.e. when the model is good enough).

    It must be used with the ``EvalCallback``.

    :param reward_threshold:  Minimum expected reward per episode
        to stop training.
    :param verbose:
    """

    def __init__(self, reward_threshold: float, verbose: int = 0):
        super(StopTrainingOnRewardThreshold, self).__init__(verbose=verbose)
        self.reward_threshold = reward_threshold

    def _on_step(self) -> bool:
        assert self.parent is not None, "``StopTrainingOnMinimumReward`` callback must be used " "with an ``EvalCallback``"
        # Convert np.bool to bool, otherwise callback() is False won't work
        continue_training = bool(self.parent.best_mean_reward < self.reward_threshold)
        if self.verbose > 0 and not continue_training:
            print(
                f"Stopping training because the mean reward {self.parent.best_mean_reward:.2f} "
                f" is above the threshold {self.reward_threshold}"
            )
        return continue_training


class EveryNTimesteps(EventCallback):
    """
    Trigger a callback every ``n_steps`` timesteps

    :param n_steps: Number of timesteps between two trigger.
    :param callback: Callback that will be called
        when the event is triggered.
    """

    def __init__(self, n_steps: int, callback: BaseCallback):
        super(EveryNTimesteps, self).__init__(callback)
        self.n_steps = n_steps
        self.last_time_trigger = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.num_timesteps
            return self._on_event()
        return True


class VideoCallback(BaseCallback):
    # Passed env should have gym Monitor wrapper
    # This enables wandb to capture the video
    def __init__(self, env, video_freq, verbose=0):
        super(VideoCallback, self).__init__(verbose)
        self.env = env
        self.video_freq = video_freq
        self.last_time_trigger = 0

    def _save_video(self):
        print("Making video")
        policy = self.model.policy
        mean_reward, std_reward = evaluate_policy(
                policy, self.env, n_eval_episodes=2, deterministic=False
        )

    def _on_step(self):
        if (self.model.num_timesteps - self.last_time_trigger) >= self.video_freq:
            self.last_time_trigger = self.model.num_timesteps
            self._save_video()

    def _on_training_end(self):
        self.env.close()


class StopTrainingOnMaxEpisodes(BaseCallback):
    """
    Stop the training once a maximum number of episodes are played.

    For multiple environments presumes that, the desired behavior is that the agent trains on each env for ``max_episodes``
    and in total for ``max_episodes * n_envs`` episodes.

    :param max_episodes: Maximum number of episodes to stop training.
    :param verbose: Select whether to print information about when training ended by reaching ``max_episodes``
    """

    def __init__(self, max_episodes: int, verbose: int = 0):
        super(StopTrainingOnMaxEpisodes, self).__init__(verbose=verbose)
        self.max_episodes = max_episodes
        self._total_max_episodes = max_episodes
        self.n_episodes = 0

    def _init_callback(self):
        # At start set total max according to number of envirnments
        self._total_max_episodes = self.max_episodes * self.training_env.num_envs

    def _on_step(self) -> bool:
        # Checking for both 'done' and 'dones' keywords because:
        # Some models use keyword 'done' (e.g.,: SAC, TD3, DQN, DDPG)
        # While some models use keyword 'dones' (e.g.,: A2C, PPO)
        done_array = np.array(self.locals.get("done") if self.locals.get("done") is not None else self.locals.get("dones"))
        self.n_episodes += np.sum(done_array).item()

        continue_training = self.n_episodes < self._total_max_episodes

        if self.verbose > 0 and not continue_training:
            mean_episodes_per_env = self.n_episodes / self.training_env.num_envs
            mean_ep_str = (
                f"with an average of {mean_episodes_per_env:.2f} episodes per env" if self.training_env.num_envs > 1 else ""
            )

            print(
                f"Stopping training with a total of {self.num_timesteps} steps because the "
                f"{self.locals.get('tb_log_name')} model reached max_episodes={self.max_episodes}, "
                f"by playing for {self.n_episodes} episodes "
                f"{mean_ep_str}"
            )
        return continue_training

# ==========================================================================
# Custom Callbacks
# ==========================================================================

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True


def save_dict_as_pkl(dic, save_dir, name=None):
    if name is not None:
        save_dir = os.path.join(save_dir, name+".pkl")
    with open(save_dir, 'wb') as out:
        pickle.dump(dic, out, protocol=pickle.HIGHEST_PROTOCOL)

class SaveRolloutsComplexCallback(BaseCallback):
    """
    Saves rollouts on exponential schedule till the difference between
    saving points is less than 500,000.
    After that, saves rollout every 500,000 timsteps.
    """
    def __init__(self, start_saving_at: int,
                sampling_env: gym.Env, save_path: str,
                k:int = 5, device: str='cpu', verbose: int =1):
        super(SaveRolloutsComplexCallback, self).__init__(verbose)
        self.next_target = start_saving_at
        self.env = sampling_env
        self.k = k
        self.device = device
        self.save_path = os.path.join(save_path, 'rollouts')
        os.mkdir(self.save_path)

    def _on_step(self):
        if self.model.num_timesteps >= self.next_target:
            self.next_target = min(self.next_target*2, self.next_target+500000)
            self._save_trajs()

    def _save_trajs(self):
        save_dir = os.path.join(self.save_path, str(self.model.num_timesteps).zfill(7))
        policy = self.model.policy
        os.mkdir(save_dir)
        for i in range(self.k):
            obs = self.env.reset()
            done = False
            traj = {"obs":[], "next_obs": [], "acs": [], "rewards": [], "dones": []}
            while not done:
                action = policy._predict(
                        (th.from_numpy(obs).to(self.device)),
                        deterministic=False).detach().cpu().numpy()
                next_obs, reward, done, infos = self.env.step(action)
                traj["obs"].append(np.array(obs))
                traj["next_obs"].append(np.array(next_obs))
                traj["acs"].append(np.array(action))
                traj["rewards"].append(np.array(reward))
                traj["dones"].append(np.array(done))
                obs = next_obs
            save_dict_as_pkl(traj, save_dir, str(i).zfill(3))


class SaveRolloutsEveryNStepsCallback(BaseCallback):
    """
    Saves 'k' number of trajectories every 'n_steps'.
    """
    def __init__(self, n_steps: int, k: int, sampling_env: gym.Env, save_path: str, 
                 device: str='cpu', verbose: int =1):
        super(SaveRolloutsEveryNStepsCallback, self).__init__(verbose)
        self.n_steps = n_steps
        self.env = sampling_env
        self.last_time_trigger = 0
        self.k = k
        self.device = device
        self.save_path = os.path.join(save_path, 'rollouts')
        os.mkdir(self.save_path)

    def _on_step(self) -> bool:
        if (self.model.num_timesteps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.model.num_timesteps
            self._save_trajs()

    def _save_trajs(self):
        save_dir = os.path.join(self.save_path, str(self.model.num_timesteps).zfill(7))
        policy = self.model.policy
        os.mkdir(save_dir)
        for i in range(self.k):
            obs = self.env.reset()
            done = False
            traj = {"obs":[], "next_obs": [], "acs": [], "rewards": [], "dones": []}
            while not done:
                action = policy._predict(
                        (th.from_numpy(obs).to(self.device)),
                        deterministic=False).detach().cpu().numpy()
                next_obs, reward, done, infos = self.env.step(action)
                traj["obs"].append(np.array(obs))
                traj["next_obs"].append(np.array(next_obs))
                traj["acs"].append(np.array(action))
                traj["rewards"].append(np.array(reward))
                traj["dones"].append(np.array(done))
                obs = next_obs
            save_dict_as_pkl(traj, save_dir, str(i).zfill(3))

         

