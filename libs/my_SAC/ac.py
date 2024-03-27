from collections import deque

import numpy as np
import torch
from torch import nn, optim
from torch.distributions.normal import Normal
from typing_extensions import override


class ReplayBuffer:
    def __init__(self, size: int) -> None:
        """Creates Replay buffer.

        Parameters
        ----------
        size:
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._maxsize = size
        self._storage = deque([], self._maxsize)

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)
        self._storage.append(data)

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return (
            np.array(obses_t),
            np.array(actions),
            np.array(rewards),
            np.array(obses_tp1),
            np.array(dones),
        )

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """

        idx = np.random.choice(
            np.arange(0, len(self._storage)), replace=True, size=batch_size
        )

        obs_batch = []
        act_batch = []
        rew_batch = []
        next_obs_batch = []
        done_mask = []
        for i in idx:
            obs_batch.append(self._storage[i][0])
            act_batch.append(self._storage[i][1])
            rew_batch.append(self._storage[i][2])
            next_obs_batch.append(self._storage[i][3])
            done_mask.append(self._storage[i][4])
        obs_batch = np.array(obs_batch)
        act_batch = np.array(act_batch)
        rew_batch = np.array(rew_batch)
        next_obs_batch = np.array(next_obs_batch)
        done_mask = np.array(done_mask)
        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask


class Actor(nn.Module):
    def __init__(self, dim_state, dim_action, hidden_dim=64) -> None:
        super().__init__()

        self.dim_state = dim_state
        self.dim_action = dim_action
        self.hidden_dim1 = hidden_dim

        self.layer_one = nn.Sequential(
            nn.Linear(self.dim_state, self.hidden_dim1),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_dim1, self.hidden_dim1),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim1, self.dim_action * 2),
            nn.LeakyReLU(0.01),
        )

    @override
    def forward(self, state):
        """
        Returns Math Expectation and Standart Deviation
        of the Random variable distribution of each action component
        """
        answer = self.layer_one(state)
        # answer = self.forward(state)
        mu, std = answer[: self.dim_action], answer[self.dim_action :]
        predicted_gauss = Normal(mu, std)

        sample = predicted_gauss.sample()
        prob = predicted_gauss.log_prob(sample)
        return sample, prob

    # def get_action(self, state):
        
    #     return sample, prob


class Critic(nn.Module):
    def __init__(self, dim_state) -> None:
        super().__init__()

        self.dim_state = dim_state
        self.hidden_dim1 = 64

        self.layer_one = nn.Sequential(
            nn.Linear(self.dim_state, self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, 1),
        )

    @override
    def forward(self, state):
        x = self.layer_one(state)
        return x


class ActorCritic:
    def __init__(self, dim_state, dim_action):
        self.actor = Actor(dim_state, dim_action)
        self.critic = Critic(dim_state)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=0.001)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=0.001)

    def train_step(self, gamma, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        value = self.critic(state)

        # value_const = value.detach()

        # Update Actor
        actions, probs = self.actor(state)
        actor_loss = - probs * value
        # ? actor_loss.zero_grad()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        # Actor weights updated

        # Calculate TD target δ
        with torch.no_grad():
            td_error = reward + gamma * self.critic(next_state) - value

        # Update Critic
        critic_loss = - td_error * value
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

    def train(self, env, num_epochs):
        # init_state = torch.tensor(state, dtype=torch.float32)
        # init_next_state = torch.tensor(next_state, dtype=torch.float32)
        # init_action = torch.tensor(action, dtype=torch.float32)
        # init_reward = torch.tensor(reward, dtype=torch.float32)
        # init_done = torch.tensor(done, dtype=torch.float32)
        for epoch in range(num_epochs):
            self.train_step(...)




# Assuming you have environment and state_dim defined
env = your_environment_creation_function()
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

actor_critic_agent = ActorCritic(num_actions)


# Main training loop
def train_actor_critic(env, actor_critic, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            state = np.reshape(state, [1, -1])  # Reshape to (1, state_dim)
            action_probs = actor_critic.actor(torch.tensor(state, dtype=torch.float32))
            action = np.random.choice(
                np.arange(len(action_probs.detach().numpy()[0])),
                p=action_probs.detach().numpy()[0],
            )

            next_state, reward, done, _ = env.step(action)

            actor_critic.train_step(
                state, torch.tensor(action), reward, next_state, done
            )

            state = next_state


# Example usage
train_actor_critic(env, actor_critic_agent)


# Main training loop
def train_actor_critic(env, actor_critic, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, -1])  # Reshape to (1, state_dim)

        while True:
            # Select action from the Actor
            action_probs = actor_critic.actor(state, training=False)
            action = np.random.choice(
                np.arange(len(action_probs[0])), p=action_probs.numpy()[0]
            )

            # Take the selected action
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, -1])  # Reshape to (1, state_dim)

            # Train the Actor-Critic
            actor_critic.train_step(
                state, tf.one_hot(action, env.action_space.n), reward, next_state, done
            )

            if done:
                break

            state = next_state


# Example usage:
# env = your_environment_creation_function()
# actor_critic_agent = ActorCritic(num_actions=env.action_space.n)
# train_actor_critic(env, actor_critic_agent, num_episodes=1000)
