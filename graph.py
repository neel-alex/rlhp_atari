import json

import matplotlib.pyplot as plt

run = 64

with open(f"results/dqfp/{run}/metrics.json") as f:
    data = json.load(f)

train_loss = data['training.loss']['values']
train_epochs = data['training.loss']['steps']
test_loss = data['test.loss']['values']
test_epochs = data['test.loss']['steps']
eval_reward = data['test.return']['values']
eval_epochs = data['test.return']['steps']

fig, host = plt.subplots(figsize=(8, 5))
par = host.twinx()

host.set_xlim(min(train_epochs), max(train_epochs) + 1)
host.set_ylim(0, max(train_loss + test_loss))
par.set_ylim(0, max(eval_reward))
host.xaxis.get_major_locator().set_params(integer=True)  # Only integer epochs

host.set_xlabel("Epochs")
host.set_ylabel("Loss")
par.set_ylabel("Reward")

color1 = "red"
color2 = "blue"
color3 = "green"

p1, = host.plot(train_epochs, train_loss, color=color1, label="Training Loss")
p2, = host.plot(test_epochs, test_loss, color=color2, label="Test Loss")
p3, = par.plot(eval_epochs, eval_reward, color=color3, label="Average Reward (5 episodes)")

lns = [p1, p2, p3]
host.legend(handles=lns, loc='best')

par.yaxis.label.set_color(p3.get_color())

plt.title("Losses and returns over the course of training DQfD with only imitation on Enduro.")
plt.savefig(f"record/{run}.png", bbox_inches='tight')
