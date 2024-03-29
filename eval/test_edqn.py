from utils.env_utils import make_atari_env
from utils.dqn_utils import ExpertDQN, DuelingDQNPolicy


def eval_episode(m, e):
    reward = 0
    obs = e.reset()
    done = False
    while not done:
        action, _ = m.predict(obs, deterministic=True)
        obs, rew, done, info = e.step(action)
        done = done[0]
        if done:
            episode_infos = info[0].get("episode")
            if episode_infos is not None:
                reward += episode_infos['r']
            if info[0]['lives'] != 0:
                obs = e.reset()
                done = False
    return reward


# 53 pretrained edqn
run_number = 62
num_episodes = 10

model_path = f"results/edqn/{run_number}/final_policy.ckpt"

env = make_atari_env("EnduroNoFrameskip-v4")
env.seed(4)
model = ExpertDQN(DuelingDQNPolicy,
                  env,
                  gamma=0.99,
                  learning_starts=0,
                  batch_size=128,
                  device='cuda',
                  seed=4,
                  verbose=1,
                  log_function=lambda *x: print('log')
                  )

# policy = model.policy
policy = DuelingDQNPolicy.load(model_path)
model.policy = policy
model.q_net = policy.q_net
model.q_net_target = policy.q_net_target

rewards = []

for i in range(num_episodes):
    print(f"Episode {i+1}...")
    reward = eval_episode(model, env)
    rewards.append(reward)

print(f"Average reward: {sum(rewards) / len(rewards)}")

