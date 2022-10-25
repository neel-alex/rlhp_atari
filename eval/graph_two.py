import json
import matplotlib.pyplot as plt

runs = 62, 63

with open(f"results/edqn/{runs[0]}/metrics.json") as f:
    d1 = json.load(f)

with open(f"results/edqn/{runs[1]}/metrics.json") as f:
    d2 = json.load(f)

def average(x):
    return sum(x)/len(x)

r1 = d1['test.return']['values']
s1 = d1['test.return']['steps']
r2 = d2['test.return']['values']
s2 = d2['test.return']['steps']
print(average(r1[-5:]))
print(average(r2[-5:]))
assert s1 == s2
s = s1

plt.xlim(min(s), max(s))

plt.xlabel("Gradient Steps")
plt.ylabel("Reward")

color1 = "blue"
color2 = "green"

p1 = plt.plot(s, r1, color=color1, label="DQfP")
p2 = plt.plot(s, r2, color=color2, label="DQN")

plt.legend()

ticks = range(0, 500001, 50000)
plt.xticks(ticks, [f"{x/100000:.1f}e5" for x in ticks], fontsize=6)
plt.savefig(f"record/tmp_{runs[0]}{runs[1]}.png", bbox_inches='tight')

