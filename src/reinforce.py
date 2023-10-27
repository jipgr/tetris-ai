import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import matplotlib.pyplot as plt

class NNPolicy(nn.Module):

    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(1,3,3,1,1), # 1x20x10 -> 3x20x10
            nn.ReLU(),
            nn.Conv2d(3,3,3), # 3x20x10 -> 3x18x8
            nn.ReLU(),
            nn.Conv2d(3,3,5), # 3x18x8 -> 3x14x4
            nn.MaxPool2d(2), # 3x7x2
            nn.Flatten(), # 42
            nn.Linear(42,3)
        )
        # self.model = nn.Sequential(
        #     nn.Flatten(), # 42
        #     nn.Linear(200,128),
        #     nn.ReLU(),
        #     nn.Linear(128,3)
        # )

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Args:
            x: input tensor (first dimension is a batch dimension)

        Return:
            Probabilities of performing all actions in given input states x. Shape: batch_size x action_space_size
        """
        return F.softmax(self.model(x), dim=-1)

    def get_probs(self, obs, actions):
        """
        This function takes a tensor of states and a tensor of actions and returns a tensor that contains
        a probability of perfoming corresponding action in all states (one for every state action pair).

        Args:
            obs: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: batch_size x 1

        Returns:
            A torch tensor filled with probabilities. Shape: batch_size x 1.
        """
        return torch.gather(self(obs), -1, actions)

    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.

        Args:
            obs: state as a tensor. Shape: 1 x obs_dim or obs_dim

        Returns:
            An action (int).
        """
        return torch.multinomial(self(obs),1).item()

# @torch.no_grad()
def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function as tensors.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of tensors (states, actions, rewards, dones). All tensors should have same first dimension and
        should have dim=2. This means that vectors of length N (states, rewards, actions) should be Nx1.
        Hint: Do not include the state after termination in states.
    """
    states = []
    actions = []
    rewards = []
    dones = []

    done = False

    # reset env
    S,_ = env.reset()
    S = torch.FloatTensor(S).reshape(1,1,20,10)

    # Run for an episode
    while not done:

        # Sample a ~ pi
        A = policy.sample_action(S)

        # Save current state
        states.append(S)
        actions.append(A)

        # Take action
        S, R, done, _, _ = env.step(A)
        S = torch.FloatTensor(S).reshape(1,1,20,10)

        # Note rewards
        rewards.append(R)
        dones.append(done)

    # Convert all to tensors of shape N,dim where N is length of episode
    # and dim=2 for states and dim=1 for actions,rewards,dones
    states = torch.cat(states)
    actions = torch.tensor(actions, dtype=torch.long).reshape((-1,1))
    rewards = torch.tensor(rewards, dtype=torch.float).reshape((-1,1))
    dones = torch.tensor(dones, dtype=torch.bool).reshape((-1,1))

    return states, actions, rewards, dones


def compute_reinforce_loss(policy, episode, discount_factor):
    """
    Computes reinforce loss for given episode.

    Args:
        policy: A policy which allows us to get probabilities of actions in states with its get_probs method.

    Returns:
        loss: reinforce loss
    """
    # Compute the reinforce loss
    # Make sure that your function runs in LINEAR TIME
    # Note that the rewards/returns should be maximized
    # while the loss should be minimized so you need a - somewhere

    # unpack episode
    states,actions,rewards,dones = episode

    # Set up G_t with G_T = R_T
    Gt = torch.zeros_like(rewards)
    Gt[-1,0] = rewards[-1,0]

    # t=T-1, ..., 0
    for t in range(Gt.shape[0]-2, -1, -1):
        # G_t = R_t + gamma * G_t+1
        Gt[t,0] = rewards[t,0] + discount_factor * Gt[t+1,0]

    # Get pi(a|s)
    probs = policy.get_probs(states, actions)

    return -(probs.log() * Gt).sum(), Gt[0,0].item()

def run_episodes_policy_gradient(policy, env, num_episodes, discount_factor, learn_rate,
                                 sampling_function=sample_episode):
    optimizer = optim.Adam(policy.parameters(), learn_rate)

    scores = []
    losses = []
    runningGain = 0.
    runningLoss = 0.
    runningLength = 0.
    for i in range(num_episodes):

        # Simulate episode
        episode = sampling_function(env, policy)
        runningLength += episode[0].shape[0] / 10

        # Compute loss for pi in this episode
        loss, G = compute_reinforce_loss(policy, episode, discount_factor)
        runningGain += G/10
        runningLoss += loss.item()/10

        # Backwards pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("Episode {:4} finished after {:4.0f} steps, average gain {:6.2f}, average loss {:8.2e}"
                .format(i, runningLength, runningGain, runningLoss))
            runningGain = 0.
            runningLoss = 0.
            runningLength = 0.
            plt.imsave(f"images/epi{i}.png", env.render(),cmap="gray")
        scores.append(G)
        losses.append(loss.item())

    return scores, losses
