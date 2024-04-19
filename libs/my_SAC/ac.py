from collections import deque
from functools import reduce

import numpy as np
import torch
from torch import nn, optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from typing_extensions import override

writer = SummaryWriter()


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
        mu, std = answer[:, : self.dim_action], answer[:, self.dim_action :]
        torch.clamp_(std, min=1e-6, max=1)
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
    def __init__(self, env):
        self.env = env
        dim_state = reduce(lambda x, y: x * y, self.env.observation_space.shape)
        dim_action = reduce(lambda x, y: x * y, self.env.action_space.shape)
        self.actor = Actor(dim_state, dim_action)
        self.critic = Critic(dim_state)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=0.001)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=0.001)

        self.step_n = 0
        self.reward = []

    def train_step(self, gamma, state, action, reward, next_state):
        self.step_n += 1

        # state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        # done = torch.tensor(done, dtype=torch.float32)

        value = self.critic(state)

        # value_const = value.detach()

        # Update Actor
        actions, probs = self.actor(state)
        actor_loss = torch.sum(-probs * value.detach())
        # ? actor_loss.zero_grad()
        self.optimizer_actor.zero_grad()
        writer.add_scalar("Loss/ActorLoss", actor_loss, self.step_n)
        actor_loss.backward(retain_graph=False)
        self.optimizer_actor.step()
        # Actor weights updated
        # value.grad.zero_()

        # Calculate TD target δ
        with torch.no_grad():
            td_error = (
                reward + gamma * self.critic(next_state).detach() - value.detach()
            )

        # Update Critic
        critic_loss = -td_error * value
        self.optimizer_critic.zero_grad()
        writer.add_scalar("Loss/CriticLoss", critic_loss, self.step_n)
        critic_loss.backward()
        self.optimizer_critic.step()

    def train(self, num_epochs):
        # init_state = torch.tensor(state, dtype=torch.float32)
        # init_next_state = torch.tensor(next_state, dtype=torch.float32)
        # init_action = torch.tensor(action, dtype=torch.float32)
        # init_reward = torch.tensor(reward, dtype=torch.float32)
        # init_done = torch.tensor(done, dtype=torch.float32)
        action_info = self.env.action_space
        action_shape = action_info.shape
        action_min = action_info.low
        action_max = action_info.high
        self.reward = []

        # s0, inform = env.reset()

        for epoch in range(num_epochs):
            s0, inform = self.env.reset()
            s0 = torch.from_numpy(s0).view(1, -1).to(torch.float32)
            a0, prob_a0 = self.actor(s0)
            torch.clamp_(a0, min=-1, max=1)
            a0 = a0.view(8).numpy()
            # a0[np.greater(a0, action_max)] = action_max[np.greater(a0, action_max)]
            # a0[np.less(a0, action_min)] = action_max[np.less(a0, action_min)]
            s1, r1, terminated, truncated, inform = self.env.step(a0)
            rewardy = r1
            while not (terminated) and not (truncated):
                self.train_step(
                    gamma=0.99, state=s0, action=a0, reward=r1, next_state=s1
                )
                s0 = torch.from_numpy(s1).view(1, -1).to(torch.float32)
                a0, prob_a0 = self.actor(s0)
                torch.clamp_(a0, min=-1, max=1)
                a0 = a0.view(8).numpy()
                # big_mask = np.greater(action_max, a0)
                # small_mask = np.less(a0, action_max)
                # a0[big_mask] = action_max[big_mask]
                # a0[small_mask] = action_max[small_mask]
                s1, r1, terminated, truncated, inform = self.env.step(a0)
                rewardy += r1
            self.reward.append(rewardy)
            writer.add_scalar("Reward/train", rewardy, epoch)
            rewardy = 0
        writer.flush()

    def demonstrate(self, num_epochs):
        # init_state = torch.tensor(state, dtype=torch.float32)
        # init_next_state = torch.tensor(next_state, dtype=torch.float32)
        # init_action = torch.tensor(action, dtype=torch.float32)
        # init_reward = torch.tensor(reward, dtype=torch.float32)
        # init_done = torch.tensor(done, dtype=torch.float32)
        action_info = self.env.action_space
        action_shape = action_info.shape
        action_min = action_info.low
        action_max = action_info.high

        # s0, inform = env.reset()

        for epoch in range(num_epochs):
            s0, inform = self.env.reset()
            s0 = torch.from_numpy(s0).view(1, -1).to(torch.float32)
            a0, prob_a0 = self.actor(s0)
            torch.clamp_(a0, min=-1, max=1)
            a0 = a0.view(8).numpy()
            # a0[np.greater(a0, action_max)] = action_max[np.greater(a0, action_max)]
            # a0[np.less(a0, action_min)] = action_max[np.less(a0, action_min)]
            s1, r1, terminated, truncated, inform = self.env.step(a0)
            while not (terminated) and not (truncated):
                s0 = torch.from_numpy(s1).view(1, -1).to(torch.float32)
                a0, prob_a0 = self.actor(s0)
                print(a0)
                torch.clamp_(a0, min=-1, max=1)
                a0 = a0.view(8).numpy()
                # a0[np.greater(a0, action_max)] = action_max[np.greater(a0, action_max)]
                # a0[np.less(a0, action_max)] = action_max[np.less(a0, action_max)]
                s1, r1, terminated, truncated, inform = self.env.step(a0)
