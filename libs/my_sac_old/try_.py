import numpy as np
import torch
from jaxtyping import Float


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.states: Float[torch.Tensor, "{self.capacity} {self.state_dim}"] = (
            torch.zeros((capacity, state_dim), dtype=torch.float32)
        )

        self.actions: Float[torch.Tensor, "{self.capacity} {self.action_dim}"] = (
            torch.zeros((capacity, action_dim), dtype=torch.float32)
        )

        self.rewards: Float[torch.Tensor, "{self.capacity} 1"] = torch.zeros(
            (capacity, 1), dtype=torch.float32
        )
        self.next_states: Float[torch.Tensor, "{self.capacity} {self.state_dim}"] = (
            torch.zeros((capacity, state_dim), dtype=torch.float32)
        )

        self.masks: Float[torch.Tensor, "{self.capacity} 1"] = torch.zeros(
            (capacity, 1), dtype=torch.float32
        )

        self.pointer = 0
        self.size = 0

    def push(
        self,
        state: Float[torch.Tensor, "{self.capacity} {self.state_dim}"],
        action: Float[torch.Tensor, "{self.capacity} {self.action_dim}"],
        reward: Float[torch.Tensor, "{self.capacity} 1"],
        next_state: Float[torch.Tensor, "{self.capacity} {self.state_dim}"],
        done: bool,
    ) -> None:
        self.states[self.pointer] = torch.tensor(state, dtype=torch.float32)
        self.actions[self.pointer] = torch.tensor(action, dtype=torch.float32)
        self.rewards[self.pointer] = torch.tensor(reward, dtype=torch.float32)
        self.next_states[self.pointer] = torch.tensor(next_state, dtype=torch.float32)
        self.masks[self.pointer] = torch.tensor(1.0 - done, dtype=torch.float32)

        self.pointer = (self.pointer) % self.capacity  # ? self.pointer + 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[
        Float[torch.Tensor, "{self.capacity} {self.state_dim}"],
        Float[torch.Tensor, "{self.capacity} {self.action_dim}"],
        Float[torch.Tensor, "{self.capacity} 1"],
        Float[torch.Tensor, "{self.capacity} {self.state_dim}"],
        Float[torch.Tensor, "{self.capacity} 1"],
    ]:
        indices = np.random.choice(self.size, batch_size, replace=False)

        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices]
        batch_masks = self.masks[indices]

        return (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_masks,
        )


"""
# Example usage:
buffer_capacity = 100000
state_dim =  observation_space_dim  # Replace with the actual dimension of your state space
action_dim = action_space_dim  # Replace with the actual dimension of your action space
batch_size = 64

replay_buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim)

# Example of how to add a transition to the replay buffer
state = env.reset()
action = np.random.rand(action_dim)
next_state, reward, done, _ = env.step(action)
replay_buffer.push(state, action, reward, next_state, done)

# Example of how to sample a batch from the replay buffer
batch_states, batch_actions, batch_rewards, batch_next_states, batch_masks = replay_buffer.sample(batch_size)
"""
