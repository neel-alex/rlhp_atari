import json

import matplotlib.pyplot as plt

run = 81

with open(f"results/edqn/{run}/metrics.json") as f:
    data = json.load(f)

def average(x):
    return sum(x)/len(x)

train_loss = data['train.loss']['values']
train_steps = data['train.loss']['steps']
eval_reward = data['test.return']['values']
print(average(eval_reward[-5:]))
eval_steps = data['test.return']['steps']

print(eval_reward)

if train_loss[0] < train_loss[1]:
    train_loss[0] = train_loss[0] * 2

fig, host = plt.subplots(figsize=(8, 5))
par = host.twinx()

host.set_xlim(min(train_steps), max(train_steps) + 1)
host.set_ylim(0, 1.2)
par.set_ylim(0, 150)
host.xaxis.get_major_locator().set_params(integer=True)  # Only integer epochs

host.set_xlabel("Gradient Steps")
host.set_ylabel("Loss")
par.set_ylabel("Reward")

color1 = "red"
color2 = "blue"
color3 = "green"

p1, = host.plot(train_steps, train_loss, color=color1, label="Training Loss")
p3, = par.plot(eval_steps, eval_reward, color=color3, label="Average Reward (5 episodes)")

lns = [p1, p3]
host.legend(handles=lns, loc='best')

par.yaxis.label.set_color(p3.get_color())

ticks = train_steps[::5] + [500000]
plt.xticks(ticks, [f"{x/100000:.1f}e5" for x in ticks], fontsize=6)
plt.savefig(f"record/edqn_{run}.png", bbox_inches='tight')
