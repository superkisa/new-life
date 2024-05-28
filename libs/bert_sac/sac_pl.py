from collections.abc import Iterator

import gymnasium as gym
import lightning as L
import numpy as np
import torch
from jaxtyping import Int64
from stable_baselines3.common.buffers import ReplayBuffer
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from typing_extensions import override

from libs.bert_sac.models import Actor, SoftQNetwork


class RLDataset(IterableDataset):
    """Contains the ReplayBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[tuple]:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]


class LitAutoEncoder(L.LightningModule):
    @override
    def __init__(
        self,
        actor_net: type[Actor],
        critic_net: type[SoftQNetwork],
        envs: gym.vector.VectorEnv,
        *,
        alpha: float,
        num_struct_elements: int,
        attention_mask: Int64[torch.Tensor, "..."],
        components_mask: Int64[torch.Tensor, "..."],
        replay_buffer_size: int = 1_000_000,
        replay_buffer_batch_size: int = 256,
        dataloader_batch_size: int = 16,
        learning_starts: int = 5_000,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_frequency: int = 2,
        target_network_frequency: int = 1,
        q_lr: float = 1e-3,
        policy_lr: float = 3e-4,
        seed: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["actor_net", "critic_net", "envs"])
        self.automatic_optimization = False  # ? enable manual optimization

        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.seed = seed
        self.learning_starts = learning_starts
        self.replay_buffer_batch_size = replay_buffer_batch_size
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.q_lr = q_lr
        self.policy_lr = policy_lr
        self.dataloader_batch_size = dataloader_batch_size

        # env setup
        self.envs = envs
        assert isinstance(
            self.envs.single_action_space, gym.spaces.Box
        ), "only continuous action space is supported"

        self.actor = actor_net(
            envs,
            num_struct_elements=num_struct_elements,
            att_mask=attention_mask,
            components_mask=components_mask,
        )
        self.qf1 = critic_net(
            envs,
            num_struct_elements=num_struct_elements,
            att_mask=attention_mask,
            components_mask=components_mask,
        )
        self.qf2 = critic_net(
            envs,
            num_struct_elements=num_struct_elements,
            att_mask=attention_mask,
            components_mask=components_mask,
        )
        self.qf1_target = critic_net(
            envs,
            num_struct_elements=num_struct_elements,
            att_mask=attention_mask,
            components_mask=components_mask,
        )
        self.qf2_target = critic_net(
            envs,
            num_struct_elements=num_struct_elements,
            att_mask=attention_mask,
            components_mask=components_mask,
        )
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.envs.single_observation_space.dtype = np.float32  # type: ignore
        self.rb = ReplayBuffer(
            replay_buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            self.device,
            handle_timeout_termination=False,
        )
        # TRY NOT TO MODIFY: start the game
        self.obs, _ = self.envs.reset(seed=self.seed)

    @override
    def training_step(self, batch, batch_idx):
        q_optimizer, actor_optimizer = self.optimizers()  # type: ignore
        # opt.zero_grad()  # type: ignore
        # loss = self.compute_loss(batch)
        # self.manual_backward(loss)
        # opt.step()  # type: ignore
        # return loss
        # ALGO LOGIC: put action logic here
        if batch_idx < self.learning_starts:
            actions = np.array(
                [self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)]
            )
        else:
            actions, _, _ = self.actor.get_action(torch.Tensor(self.obs).to(self.device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        self.rb.add(self.obs, real_next_obs, actions, rewards, terminations, infos)  # type: ignore

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        self.obs = next_obs

        # ! ALGO LOGIC: training.
        if batch_idx > self.learning_starts:
            data = self.rb.sample(self.replay_buffer_batch_size)
            # region #! critic optimize step
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = self.actor.get_action(
                    data.next_observations
                )
                qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)

                # * H(P) = 𝔼[- log P]
                entropy = -self.alpha * next_state_log_pi  # alpha is a step

                # * Q_next = min(Q_1_target, Q_2_target) + alpha * H
                future_reward = torch.min(qf1_next_target, qf2_next_target) + entropy

                # * Q_now = reword + gamma * Q_next
                cum_reward = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * (
                    future_reward
                ).view(-1)  # ? shape (?,)

            qf1_a_values = self.qf1(data.observations, data.actions).view(-1)  # Q_now from qf1 NN
            qf2_a_values = self.qf2(data.observations, data.actions).view(-1)  # Q_now from qf2 NN

            # Comparing Q_now from qf1 NN and Q_now counted with reword + gamma * Q_next
            qf1_loss = F.mse_loss(qf1_a_values, cum_reward)
            # Comparing Q_now from qf2 NN and Q_now counted with reword + gamma * Q_next
            qf2_loss = F.mse_loss(qf2_a_values, cum_reward)
            qf_loss = qf1_loss + qf2_loss

            # * optimizing the weights of qf1 and qf2 NNs
            q_optimizer.zero_grad()
            self.manual_backward(qf_loss)
            q_optimizer.step()
            # endregion

            if batch_idx % self.policy_frequency == 0:  # TD 3 Delayed update support
                # * compensate for the delay by doing 'actor_update_interval' instead of 1
                for _ in range(self.policy_frequency):
                    pi, log_pi, _ = self.actor.get_action(data.observations)
                    qf1_pi = self.qf1(data.observations, pi)
                    qf2_pi = self.qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    self.manual_backward(actor_loss)
                    actor_optimizer.step()

            # update the target networks
            if batch_idx % self.target_network_frequency == 0:
                for param, target_param in zip(
                    self.qf1.parameters(), self.qf1_target.parameters(), strict=False
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.qf2.parameters(), self.qf2_target.parameters(), strict=False
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

    @override
    def configure_optimizers(self):
        q_optimizer = Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.q_lr)
        actor_optimizer = Adam(list(self.actor.parameters()), lr=self.policy_lr)
        return q_optimizer, actor_optimizer

    @override
    def train_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.hparams["replay_buffer_batch_size"])
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams["dataloader_batch_size"],
        )
        return dataloader
