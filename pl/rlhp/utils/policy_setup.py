from stable_baselines3 import PPO, SAC, DQN, TD3, DDPG
import rlhp.general_utils as utils

from dqn_utils import ExpertMarginDQN, DuelingDQNPolicy

def setup_policy(train_env, config, reward_model):
    if config.rl_algo == 'ppo':
        model = PPO(
            policy=config.policy_name,
            env=train_env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            clip_range_vf=config.clip_range_vf,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            target_kl=config.target_kl,
            seed=config.seed,
            device=config.device,
            verbose=config.verbose,
            policy_kwargs=dict(net_arch=utils.get_net_arch(config))
        )
    # TODO: May be merge SAC,DDPG and TD3 into one algo?
    elif config.rl_algo == 'sac':
        model = SAC(
            policy=config.policy_name,
            env=train_env,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            train_freq=config.train_freq,
            learning_starts=config.learning_starts,
            gamma=config.gamma,
            seed=config.seed,
            device=config.device,
            verbose=config.verbose,
            policy_kwargs=dict(net_arch=utils.get_net_arch(config)),
            reward_model=reward_model,
            reward_relabeling=config.reward_relabeling,
        )
    elif config.rl_algo == 'ddpg':
        model = DDPG(
            policy=config.policy_name,
            env=train_env,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            train_freq=config.train_freq,
            n_episodes_rollout=-1,
            learning_starts=config.learning_starts,
            gamma=config.gamma,
            seed=config.seed,
            device=config.device,
            verbose=config.verbose,
            policy_kwargs=dict(net_arch=utils.get_net_arch(config)),
            reward_model=reward_model,
            reward_relabeling=config.reward_relabeling,
        )
    elif config.rl_algo == 'td3':
        model = TD3(
            policy=config.policy_name,
            env=train_env,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            train_freq=config.train_freq,
            n_episodes_rollout=-1,
            policy_delay=config.policy_delay,
            learning_starts=config.learning_starts,
            gamma=config.gamma,
            seed=config.seed,
            device=config.device,
            verbose=config.verbose,
            policy_kwargs=dict(net_arch=utils.get_net_arch(config)),
            reward_model=reward_model,
            reward_relabeling=config.reward_relabeling,
        )
    elif config.rl_algo == "edqn":
        model = ExpertMarginDQN(DuelingDQNPolicy,
                                train_env,
                                replay_buffer_kwargs={
                                     "expert_observations": None,
                                     "expert_actions": None,
                                     "expert_rewards": None,
                                     "expert_next_observations": None,
                                     "reward_model": reward_model,
                                     "reward_relabeling": True,
                                    },
                                )

    return model



