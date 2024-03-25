import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from collections import deque


class ReplayBuffer:
    def __init__(self, size):
        """
        Create Replay buffer.
        Parameters
        ----------
        size: int
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

        idx = np.random.choice(np.arange(0, len(self._storage)), replace=True, size=batch_size)

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
    def __init__(self, dim_state, dim_action):
        super().__init__() #Actor, self
        # Define the layers for the Actor network
        # Example architecture: Dense -> ReLU -> Dense -> Softmax
        # Adjust the architecture based on your problem

        self.dim_state = dim_state
        self.dim_action = dim_action
        self.hidden_dim1 = 64

        self.layer_one = nn.Sequential(
            nn.Linear(self.dim_state, self.hidden_dim1),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_dim1, self.hidden_dim1),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim1, self.hidden_dim1),
            nn.LeakyReLU(0.01),
        )


        # self.input_layer = nn.Linear(self.dim_state, self.hidden_dim1)
        # self.leaky_relu1 = nn.LeakyReLU(0.1)

        # self.linear2 = nn.Linear(self.hidden_dim1, self.hidden_dim1)
        # self.leaky_relu2 = nn.LeakyReLU(0.2)

        # self.linear3 = nn.Linear(self.hidden_dim1, self.hidden_dim1)
        # self.leaky_relu3 = nn.LeakyReLU(0.1)

        # self.linear4 = nn.Linear(self.hidden_dim1, self.hidden_dim1)
        # self.leaky_relu4 = nn.LeakyReLU(0.01)

        # self.linear5 = nn.Linear(self.hidden_dim1, self.hidden_dim1)
        # self.leaky_relu5 = nn.LeakyReLU(0.1)

        # self.linear6 = nn.Linear(self.hidden_dim1, self.hidden_dim1)
        # self.leaky_relu6 = nn.LeakyReLU(0.2)

        # self.linear7 = nn.Linear(self.hidden_dim1, self.hidden_dim1)
        # self.leaky_relu7 = nn.LeakyReLU(0.3)

        # self.linear8 = nn.Linear(self.hidden_dim1, self.hidden_dim1)
        # self.leaky_relu8 = nn.LeakyReLU(0.01)

        # self.linear9 = nn.Linear(self.hidden_dim1, self.dim_action)
        # self.tanh = nn.Tanh()


    def forward(self, state):
        # Define the forward pass for the Actor
        x = self.layer_one(self)
        
        # x = self.input_layer(state)
        # x = self.leaky_relu1(x)

        # x = self.linear2(x)
        # x = self.leaky_relu2(x)

        # x = self.linear3(x)
        # x = self.leaky_relu3(x)

        # x = self.linear4(x)
        # x = self.leaky_relu4(x)
        # x = self.linear5(x)
        # x = self.leaky_relu5(x)

        # x = self.linear6(x)
        # x = self.leaky_relu6(x)

        # x = self.linear7(x)
        # x = self.leaky_relu7(x)

        # x = self.linear8(x)
        # x = self.leaky_relu8(x)

        # x = self.linear9(x)
        # x = self.tanh(x)

        return x


class Critic(nn.Module):
    def __init__(self, dim_state):
        super().__init__() #Actor, self
        # Define the layers for the Actor network
        # Example architecture: Dense -> ReLU -> Dense -> Softmax
        # Adjust the architecture based on your problem

        self.dim_state = dim_state
        self.hidden_dim1 = 64


        self.layer_one = nn.Sequential(
            nn.Linear(self.dim_state, self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, self.hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim1, self.hidden_dim1),
            nn.ReLU(),
        )


        # self.input_layer = nn.Linear(self.dim_state, self.hidden_dim1)
        # self.relu1 = nn.ReLU()

        # self.linear2 = nn.Linear(self.hidden_dim1, self.hidden_dim1)
        # self.relu2 = nn.ReLU()

        # self.linear3 = nn.Linear(self.hidden_dim1, self.hidden_dim1)
        # self.relu3 = nn.ReLU()

        # self.linear4 = nn.Linear(self.hidden_dim1, self.hidden_dim1)
        # self.relu4 = nn.ReLU()

        # self.linear5 = nn.Linear(self.hidden_dim1, self.hidden_dim1)
        # self.relu5 = nn.ReLU()

        # self.linear6 = nn.Linear(self.hidden_dim1, self.hidden_dim1)
        # self.relu6 = nn.ReLU()

        # self.linear7 = nn.Linear(self.hidden_dim1, self.hidden_dim1)
        # self.relu7 = nn.ReLU()

        # self.linear8 = nn.Linear(self.hidden_dim1, self.hidden_dim1)
        # self.relu8 = nn.ReLU()

        # self.linear9 = nn.Linear(self.hidden_dim1, self.hidden_dim1)
        # self.relu9 = nn.ReLU()


    def forward(self, state):
        # Define the forward pass for the Actor

        x = self.layer_one(self)


        # x = self.input_layer(state)
        # x = self.relu1(x)

        # x = self.linear2(x)
        # x = self.relu2(x)

        # x = self.linear3(x)
        # x = self.relu3(x)

        # x = self.linear4(x)
        # x = self.leaky_relu4(x)
        # x = self.linear5(x)
        # x = self.relu5(x)

        # x = self.linear6(x)
        # x = self.relu6(x)

        # x = self.linear7(x)
        # x = self.relu7(x)

        # x = self.linear8(x)
        # x = self.relu8(x)

        # x = self.linear9(x)
        # x = self.relu9(x)

        return x


class ActorCritic:
    def __init__(self, dim_state, dim_action):
        self.actor = Actor(dim_state, dim_action)
        self.critic = Critic(dim_state)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=0.001)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=0.001)

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        # Calculate TD target
        with torch.no_grad():
            target = reward + (1 - done) * 0.99 * self.critic(next_state)

        # Update Critic
        value = self.critic(state)
        critic_loss = nn.MSELoss()(value, target)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Update Actor
        action_probs = self.actor(state)
        chosen_action_prob = torch.sum(action_probs * action, dim=1)
        advantage = target - value
        actor_loss = -torch.log(chosen_action_prob) * advantage
        self.optimizer_actor.zero_grad()
        actor_loss.mean().backward()
        self.optimizer_actor.step()

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
            action = np.random.choice(np.arange(len(action_probs.detach().numpy()[0])), p=action_probs.detach().numpy()[0])

            next_state, reward, done, _ = env.step(action)

            actor_critic.train_step(state, torch.tensor(action), reward, next_state, done)

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
            action = np.random.choice(np.arange(len(action_probs[0])), p=action_probs.numpy()[0])

            # Take the selected action
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, -1])  # Reshape to (1, state_dim)

            # Train the Actor-Critic
            actor_critic.train_step(state, tf.one_hot(action, env.action_space.n), reward, next_state, done)

            if done:
                break

            state = next_state


# Example usage:
# env = your_environment_creation_function()
# actor_critic_agent = ActorCritic(num_actions=env.action_space.n)
# train_actor_critic(env, actor_critic_agent, num_episodes=1000)
