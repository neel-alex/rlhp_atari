import torch as th
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


# TODO: use reward model instead of environment rewards...
def large_margin_loss(action_qs, target_action, margin=1.0):
    target_indices = th.reshape(target_action.to(th.int64), target_action.shape + (1,))
    # Get Qs of expert actions
    expert_margin = th.full(action_qs.shape, margin).to("cuda:0")
    margins = th.zeros(target_indices.shape).to("cuda:0")
    expert_margin = expert_margin.scatter(1, target_indices, margins)
    margin_adjusted_qs = action_qs + expert_margin
    best_qs, _ = th.max(margin_adjusted_qs, 1)

    expert_qs = th.gather(action_qs, -1, target_indices)
    expert_qs = expert_qs.reshape(best_qs.shape)
    return th.mean(best_qs - expert_qs)


def train_policy(model, train_loader, epochs, device="auto", learning_rate=1e-3,
                 eps=1e-8, weight_decay=0, scheduler_gamma=1.0, experiment=None,
                 test_interval=10, test_loader=None, eval_interval=10,
                 eval_env=None, save_interval=50, observer=None, verbose=True):
    policy = model.policy.to(device)

    optimizer = optim.Adam(policy.parameters(), lr=learning_rate, eps=eps, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    policy.train()

    for epoch in range(epochs):
        if test_interval is not None and epoch % test_interval == 0 and test_loader is not None:
            test_policy(policy, test_loader, device=device, experiment=experiment, epoch=epoch, verbose=verbose)
        if eval_interval is not None and epoch % eval_interval == 0 and eval_env is not None:
            eval_model(model, eval_env, experiment=experiment, epoch=epoch)
        if save_interval is not None and epoch % save_interval == 0 and observer is not None:
            model.save(observer.dir + f"/epoch_{epoch}_policy_checkpoint")

        epoch_loss = th.tensor(0.0).to(device)
        for state, action, rewards, next_states in train_loader:
            state = state.to(device)
            action = action.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)

            optimizer.zero_grad()
            action_qs = policy.q_net(state)

            loss = large_margin_loss(action_qs, action)
            """
            # 1-step Q loss
            one_step_states = next_states[:, 0]

            one_step_qs, _ = th.max(th.mul(policy.q_net_target(one_step_states), discount), 1)
            one_step_targets = rewards[:, 0] + one_step_qs
            loss = loss + criterion(expert_qs, one_step_targets)

            # n-step Q loss
            n_step_states = next_states[:, -1]
            n_step_qs, _ = th.max(th.mul(policy.q_net_target(n_step_states), discount ** rewards.shape[1]), 1)
            discounted_rewards = sum([discount ** i * reward for i, reward in enumerate(rewards.T)])
            n_step_targets = discounted_rewards + n_step_qs
            loss = loss + criterion(expert_qs, n_step_targets)

            # Regularization loss
            # Not necessary to implement since Pytorch Adam optimizer uses L2 reg
            #   for weight decay??
            #   https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
            """
            epoch_loss += loss
            loss.backward()
            optimizer.step()

        scheduler.step()

        epoch_loss = epoch_loss.item() / len(train_loader.dataset)
        if experiment:
            experiment.log_scalar("training.loss", epoch_loss, step=epoch)
        if verbose:
            print(f"Training loss ({epoch if epoch is not None else ''}): {epoch_loss:4f}")

    if test_interval is not None and test_loader is not None:
        test_policy(policy, test_loader, device=device, experiment=experiment, epoch=epochs, verbose=verbose)
    if eval_interval is not None and eval_env is not None:
        eval_model(model, eval_env, experiment=experiment, epoch=epochs)


def test_policy(policy, test_loader, device="auto", experiment=None, epoch=None, verbose=True):
    policy.eval()
    test_loss = th.tensor(0.0).to(device)
    with th.no_grad():
        for state, action, _, _ in test_loader:
            state = state.to(device)
            action = action.to(device)
            action_qs = policy.q_net(state)

            test_loss += large_margin_loss(action_qs, action)

    test_loss /= len(test_loader.dataset)
    if experiment is not None:
        experiment.log_scalar("test.loss", test_loss.item(), step=epoch)
    if verbose:
        print(f"Test loss ({epoch if epoch is not None else ''}): {test_loss:.4f}")


def eval_model(model, env, num_episodes=5, experiment=None, epoch=None, verbose=True):
    rewards = []
    for i in range(num_episodes):
        rewards.append(eval_episode(model, env))
    avg_reward = sum(rewards) / len(rewards)
    if experiment is not None:
        experiment.log_scalar("test.return", avg_reward, step=epoch)
    if verbose:
        print(f"Average Reward ({epoch if epoch is not None else ''}): {avg_reward:.2f}")


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
