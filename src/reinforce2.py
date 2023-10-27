import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eps = np.finfo(np.float32).eps.item()

class Policy(nn.Module):
    def __init__(self, dim_in=4, dim_hidden=128, num_actions=2):
        super(Policy, self).__init__()
        # self.model = nn.Sequential(
        #     nn.Linear(dim_in, dim_hidden),
        #     nn.ReLU(),
        #     nn.Linear(dim_hidden, num_actions)
        # )
        self.model = nn.Sequential(
            nn.Conv2d(1,3,3,1,1), # 1x20x10 -> 3x20x10
            nn.ReLU(),
            nn.Conv2d(3,3,3), # 3x20x10 -> 3x18x8
            nn.ReLU(),
            nn.Conv2d(3,3,5), # 3x18x8 -> 3x14x4
            nn.MaxPool2d(2),
            nn.Flatten(), # 42
            nn.Linear(42,5),
            nn.Softmax(dim=1)
        )

    def forward(self, obs):
        return F.softmax(self.model(obs), dim=1)


    def select_action(self, state):
        probs = self(state)
        m = Categorical(probs)

        action = m.sample()
        log_prob = m.log_prob(action)

        return action.item(), log_prob


def train(
        env,
        num_episodes:int,
        lr:float,
        max_episode_length:int=3000,
        gamma:float=.99,
        log_interval:int=10
):
    policy = Policy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    avgLength = 0
    avgGain = 0
    scores = []
    for i in range(1,num_episodes+1):

        rewards = []
        log_probs = []

        action_counter = defaultdict(int)

        state, _ = env.reset()
        state = torch.from_numpy(state).float().reshape(1,1,*state.shape).to(device)

        for t in range(max_episode_length):  # Don't infinite loop while learning

            action, log_prob = policy.select_action(state)

            action_counter[action] += 1

            # print(len(env.step(action)))
            state, reward, done, _ = env.step(action)
            state = torch.from_numpy(state).float().reshape(1,1,*state.shape).to(device)

            rewards.append(reward)
            log_probs.append(log_prob)

            if done:
                break

        avgLength += t / log_interval

        gains = []
        R = 0.
        for r in rewards[::-1]:
            R = r + gamma * R
            gains.insert(0, R)

        avgGain += R / log_interval
        scores.append(R)

        gains = torch.tensor(gains).to(device)
        gains = (gains - gains.mean()) / (gains.std() + eps)

        policy_loss = []
        for log_prob, gain in zip(log_probs, gains):
            policy_loss.append( -log_prob * gain )

        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print('Episode {:5}, Average length: {:5.0f}, Average gain: {:7.2f}'.format(
                i, avgLength, avgGain))

            norm = sum(action_counter.values())
            print("\tAction dist:", ", ".join(f"{a}:{action_counter[a]/norm*100:3.0f}" for a in range(5)))

            action_counter = defaultdict(int)
            avgLength = 0.
            avgGain = 0.

            # plt.imsave(f"images/epi{i}.png", env.render(),cmap="gray")

    return scores
