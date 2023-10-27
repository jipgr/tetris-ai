from src.env import TetrisEnv
import src.reinforce
import src.reinforce2 as reinforce

import numpy as np
import matplotlib.pyplot as plt
import gym

# Smoothing function for nicer plots
def smooth(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

env = TetrisEnv(render_mode="rgb_array")

num_episodes = 5000
discount_factor = 0.99
learn_rate = 1e-8

scores = reinforce.train(
    env=env,
    num_episodes=num_episodes,
    gamma=discount_factor,
    lr=learn_rate,
    log_interval=50,
)

fig = plt.figure(1, figsize=(6,6))

ax = fig.add_subplot(111)
ax.plot(smooth(scores, 5), c='g')
ax.set_ylabel("Score")

# ax2 = ax.twinx()
# ax2.plot(smooth(losses,5), c='r')
# ax2.set_ylabel("Loss")

plt.savefig("out.png")
plt.show()
