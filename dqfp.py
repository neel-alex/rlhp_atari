import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

margin = 0.8

# TODO: use reward model instead of environment rewards...


def train_policy(model, train_loader, epochs, device="auto", discount=0.99,
                 learning_rate=1e-3, weight_decay=1e-5, scheduler_gamma=0.7):
    policy = model.policy.to(device)
    policy.train()

    criterion = nn.MSELoss()  # Almost certainly the wrong loss function... One hot targets, train logits? Unclear.
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    for epoch in range(epochs):
        for state, action, rewards, next_states in train_loader:
            state = state.to(device)
            action = action.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)

            optimizer.zero_grad()
            action_qs = policy.q_net(state)

            # Large Margin Loss
            target_indices = th.reshape(action.to(th.int64), action.shape + (1,))
            # Get Qs of expert actions
            expert_margin = th.full(action_qs.shape, margin).to("cuda:0")
            margins = th.zeros(target_indices.shape).to("cuda:0")
            expert_margin = expert_margin.scatter(1, target_indices, margins)
            margin_adjusted_qs = action_qs + expert_margin
            best_qs, _ = th.max(margin_adjusted_qs, 1)

            expert_qs = th.gather(action_qs, -1, target_indices)
            expert_qs = expert_qs.reshape(best_qs.shape)
            loss = th.mean(best_qs - expert_qs)
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
            loss.backward()
            optimizer.step()

        scheduler.step()

    return policy


def test_policy(model, test_loader, device="auto"):
    policy = model.policy.to(device)
    policy.eval()
    test_loss = 0
    flag = True
    with th.no_grad():
        for state, action, _, _ in test_loader:

            state = state.to(device)
            action = action.to(device)

            action_qs = policy.q_net(state)



            # Large Margin Loss
            target_indices = th.reshape(action.to(th.int64), action.shape + (1,))
            # Get Qs of expert actions
            expert_margin = th.full(action_qs.shape, margin).to("cuda:0")
            margins = th.zeros(target_indices.shape).to("cuda:0")
            expert_margin = expert_margin.scatter(1, target_indices, margins)
            margin_adjusted_qs = action_qs + expert_margin
            best_qs, _ = th.max(margin_adjusted_qs, 1)



            expert_qs = th.gather(action_qs, -1, target_indices)
            expert_qs = expert_qs.reshape(best_qs.shape)

            if flag:
                print(action_qs)
                print(action)
                print(target_indices.reshape(action.shape))
                print(expert_qs)
                print(best_qs)
            flag = False

            test_loss += th.mean(best_qs - expert_qs)

    test_loss /= len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}")

