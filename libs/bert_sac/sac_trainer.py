import random
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from jaxtyping import Int64
from stable_baselines3.common.buffers import ReplayBuffer
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm import tqdm

from libs.bert_sac.models import Actor, CleanRLActor, SoftQNetwork


class AntSAC:
    def __init__(
        self,
        actor_net: type[CleanRLActor],
        critic_net: type[SoftQNetwork],
        envs: gym.vector.VectorEnv,
        device: torch.device,
        *,
        # num_struct_elements: int,
        attention_mask: Int64[torch.Tensor, "..."],
        # components_mask: Int64[torch.Tensor, "..."],
        n_legs: int = 4,
        alpha: float = 0.2,
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
        torch_deterministic: bool = False,
        seed: int | None = None,
    ):
        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.seed = seed
        self.n_legs = n_legs
        self.learning_starts = learning_starts
        self.replay_buffer_batch_size = replay_buffer_batch_size
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.q_lr = q_lr
        self.policy_lr = policy_lr
        self.dataloader_batch_size = dataloader_batch_size

        self.torch_deterministic = torch_deterministic

        # TRY NOT TO MODIFY: seeding
        torch.backends.cudnn.deterministic = self.torch_deterministic
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        # env setup
        self.envs = envs
        assert isinstance(
            self.envs.single_action_space, gym.spaces.Box
        ), "only continuous action space is supported"

        self.actor = actor_net(
            envs,
            att_mask=attention_mask,
        )
        self.qf1 = critic_net(att_mask=attention_mask, num_obs=27, num_act=8)
        self.qf2 = critic_net(att_mask=attention_mask, num_obs=27, num_act=8)
        self.qf1_target = critic_net(att_mask=attention_mask, num_obs=27, num_act=8)
        self.qf2_target = critic_net(att_mask=attention_mask, num_obs=27, num_act=8)

        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self._q_optimizer = Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.q_lr
        )
        self._actor_optimizer = Adam(list(self.actor.parameters()), lr=self.policy_lr)

        self.envs.single_observation_space.dtype = np.float32  # type: ignore
        self.rb = ReplayBuffer(
            replay_buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            self.device,
            handle_timeout_termination=False,
        )

    def train(self, total_timesteps: int, finetune=False, weights_old=[]):

        if finetune:
            self.actor.load_state_dict(weights_old[0])
            self.qf1.load_state_dict(weights_old[1])
            self.qf2.load_state_dict(weights_old[2])
            self.qf1_target.load_state_dict(weights_old[1])
            self.qf2_target.load_state_dict(weights_old[2])

        date_time = datetime.now().strftime("%Y.%m.%d_%H-%M")
        env_id = self.envs.get_attr("spec")[0].id
        run_name = f"{env_id}__{self.seed}__{int(time.time())}"

        writer = SummaryWriter(f"runs/{run_name}")
        # writer.add_text(
        #     "hyperparameters",
        #     "|param|value|\n|-|-|\n%s"
        #     % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        # )

        # TRY NOT TO MODIFY: start the game
        self.obs, _ = self.envs.reset(seed=self.seed)

        for global_step in tqdm(range(total_timesteps)):
            self.training_step(None, global_step)

        # region #! on exit cleanup
        torch.save(
            self.actor.state_dict(),
            "weights/actor_" + str(self.n_legs) + "legs_" + date_time + ".pt",
        )
        torch.save(
            self.qf1.state_dict(), "weights/qf1_" + str(self.n_legs) + "legs_" + date_time + ".pt"
        )
        torch.save(
            self.qf2.state_dict(), "weights/qf2_" + str(self.n_legs) + "legs_" + date_time + ".pt"
        )
        torch.save(
            self.qf1_target.state_dict(),
            "weights/qf1_target_" + str(self.n_legs) + "legs_" + date_time + ".pt",
        )
        torch.save(
            self.qf2_target.state_dict(),
            "weights/qf2_target_" + str(self.n_legs) + "legs_" + date_time + ".pt",
        )
        self.envs.close()
        writer.close()
        # endregion

    def training_step(self, batch, batch_idx):
        q_optimizer, actor_optimizer = self.optimizers()

        # ALGO LOGIC: put action logic here
        if batch_idx < self.learning_starts:
            actions = np.array(
                [self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)]
            )
        else:
            obs_t = torch.Tensor(self.obs).to(self.device)
            actions, _, _ = self.actor.get_action(obs_t)
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
            # self.manual_backward(qf_loss)
            qf_loss.backward()
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
                    # self.manual_backward(actor_loss)
                    actor_loss.backward()
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

    def optimizers(self):
        return self._q_optimizer, self._actor_optimizer
