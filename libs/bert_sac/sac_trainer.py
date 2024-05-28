import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm import tqdm
from typing_extensions import override

from libs.bert_sac.models import CleanRLActor, SoftQNetwork


@dataclass(slots=True, frozen=True)
class AntSACConfig:
    n_legs: int = 4
    alpha: float = 0.2
    replay_buffer_size: int = 1_000_000
    replay_buffer_batch_size: int = 256
    dataloader_batch_size: int = 16
    learning_starts: int = 5_000
    gamma: float = 0.99
    tau: float = 0.005
    policy_frequency: int = 2
    target_network_frequency: int = 1
    q_lr: float = 1e-3
    policy_lr: float = 3e-4
    torch_deterministic: bool = False
    checkpoint_frequency: int = 10_000
    seed: int | None = None


@dataclass(slots=True, frozen=True)
class AntSACCheckpoint:
    step: int
    model_state_dict: dict[str, Any]
    critic_optimizer_state_dict: dict[str, Any]
    actor_optimizer_state_dict: dict[str, Any]
    critic_loss: torch.Tensor | None = None
    actor_loss: torch.Tensor | None = None


class AntSAC(torch.nn.Module):
    @override
    def __init__(
        self,
        actor_net: type[CleanRLActor],
        critic_net: type[SoftQNetwork],
        envs: gym.vector.VectorEnv,
        *,
        device: torch.device,
        attention_mask: torch.Tensor,
        artifact_path: Path | str,
        config: AntSACConfig,
        checkpoint_type: type[AntSACCheckpoint] = AntSACCheckpoint,
    ):
        super().__init__()

        self.global_step: int = 0

        self.device = device

        self.artifact_path = Path(artifact_path)
        self.config = config
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.tau = config.tau
        self.seed = config.seed
        self.n_legs = config.n_legs
        self.learning_starts = config.learning_starts
        self.replay_buffer_batch_size = config.replay_buffer_batch_size
        self.policy_frequency = config.policy_frequency
        self.target_network_frequency = config.target_network_frequency
        self.q_lr = config.q_lr
        self.policy_lr = config.policy_lr
        self.dataloader_batch_size = config.dataloader_batch_size

        self.torch_deterministic = config.torch_deterministic
        self.checkpoint_frequency = config.checkpoint_frequency
        self.checkpoint_type = checkpoint_type

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
            config.replay_buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            self.device,
            handle_timeout_termination=False,
        )

        self.writer = None

    def init_writer(self):
        env_id = self.envs.get_attr("spec")[0].id
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{env_id}__{self.seed}__{now}"

        log_path = self.artifact_path.resolve() / "runs" / run_name
        log_path.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(log_path))
        return self.writer

    def train(self, total_timesteps: int):
        # date_time = datetime.now().strftime("%Y.%m.%d_%H-%M")

        writer = self.init_writer()
        hparams = asdict(self.config)
        writer.add_hparams(hparams, {})
        # writer.add_text(
        #     "hyperparameters",
        #     "|param|value|\n|-|-|\n{}".format(
        #         "\n".join([f"|{key}|{value}|" for key, value in hparams.items()])
        #     ),
        # )

        # TRY NOT TO MODIFY: start the game
        self.obs, _ = self.envs.reset(seed=self.seed)

        for step in tqdm(range(self.global_step, total_timesteps)):
            self.global_step = step
            self.training_step(None, step)

        self.close()

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

        if "final_info" in infos:
            info = next(iter(infos["final_info"]))
            # print(f"global_step={batch_idx}, episodic_return={info['episode']['r']}")
            if self.writer:
                self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], batch_idx)
                self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], batch_idx)

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

                # * Q_now = reward + gamma * Q_next
                cum_reward = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * (
                    future_reward
                ).view(-1)  # ? shape (?,)

            qf1_a_values = self.qf1(data.observations, data.actions).view(-1)  # Q_now from qf1 NN
            qf2_a_values = self.qf2(data.observations, data.actions).view(-1)  # Q_now from qf2 NN

            # Comparing Q_now from qf1 NN and Q_now counted with reward + gamma * Q_next
            qf1_loss = F.mse_loss(qf1_a_values, cum_reward)
            # Comparing Q_now from qf2 NN and Q_now counted with reward + gamma * Q_next
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

            if batch_idx % 100 == 0 and self.writer:
                self.writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), batch_idx)
                self.writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), batch_idx)
                self.writer.add_scalar("losses/qf1_loss", qf1_loss.item(), batch_idx)
                self.writer.add_scalar("losses/qf2_loss", qf2_loss.item(), batch_idx)
                self.writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, batch_idx)
                self.writer.add_scalar("losses/actor_loss", actor_loss.item(), batch_idx)
                self.writer.add_scalar("losses/alpha", self.alpha, batch_idx)

            # save checkpoints
            if batch_idx % self.checkpoint_frequency == 0:
                print("Saving checkpoint...")
                self.save_checkpoint(step=batch_idx)
                print("Checkpoints saved.")

    def optimizers(self):
        return self._q_optimizer, self._actor_optimizer

    def save_checkpoint(
        self,
        step: int | None,
        actor_loss: torch.Tensor | None = None,
        critic_loss: torch.Tensor | None = None,
    ):
        step = step if step is not None else self.global_step
        env_id = self.envs.get_attr("spec")[0].id
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{env_id}__{self.seed}__{now}"

        save_path = self.artifact_path.resolve() / "checkpoints"
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            obj=self.checkpoint_type(
                step=step,
                model_state_dict=self.state_dict(),
                actor_optimizer_state_dict=self._actor_optimizer.state_dict(),
                critic_optimizer_state_dict=self._q_optimizer.state_dict(),
                actor_loss=actor_loss,
                critic_loss=critic_loss,
            ),
            f=save_path / (run_name + ".tar"),
        )

    def load_from_checkpoint(self, path: Path | str):
        cpt = torch.load(path)
        assert isinstance(cpt, self.checkpoint_type)
        self.global_step = cpt.step
        self.load_state_dict(cpt.model_state_dict)
        self._q_optimizer.load_state_dict(cpt.critic_optimizer_state_dict)
        self._actor_optimizer.load_state_dict(cpt.actor_optimizer_state_dict)

    def close(self):
        self.envs.close()
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()
