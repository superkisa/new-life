# docs and experiment results can be found at
# https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from gymnasium.envs.registration import EnvSpec
from stable_baselines3.common.buffers import ReplayBuffer
from torch import optim
from torch.utils.tensorboard import SummaryWriter  # type: ignore


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 25_000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = int(5e3)
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target networks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


def make_env(env_id, seed, idx, capture_video):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, video_folder="videos")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


@dataclass(slots=True, frozen=True)
class SACConfig:
    # env: gym.Env
    n_legs: int
    num_struct_elements: int
    att_mask: torch.LongTensor
    components_mask: torch.LongTensor
    finetune: bool = False

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""

    seed: int = 1
    """seed of the experiment"""

    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""

    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str | EnvSpec = "Hopper-v4"
    """the environment id of the task"""

    total_timesteps: int = 25_000
    """total timesteps of the experiments"""

    buffer_size: int = 1_000_000
    """the replay memory buffer size"""

    gamma: float = 0.99
    """the discount factor gamma"""

    tau: float = 0.005
    """target smoothing coefficient"""

    batch_size: int = 256
    """the batch size of sample from the reply memory"""

    learning_starts: int = int(5e3)
    """timestep to start learning"""

    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""

    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""

    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""

    target_network_frequency: int = 1  # also try 2
    """the frequency of updates for the target networks"""

    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    alpha: float = 0.2
    """entropy regularization coefficient"""

    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


class SAC:
    def __init__(
        self,
        config: SACConfig,
        args: Args,
    ):
        # self.env = config.env
        self.n_legs = config.n_legs
        self.num_struct_elements = config.num_struct_elements
        self.att_mask = config.att_mask
        self.components_mask = config.components_mask
        self.args = args

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.args.cuda else "cpu"
        )
        # TRY NOT TO MODIFY: seeding
        self._seed()

        # env setup
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(self.args.env_id, self.args.seed, 0, self.args.capture_video)]
        )
        assert isinstance(
            self.envs.single_action_space, gym.spaces.Box
        ), "only continuous action space is supported"

        # max_action = float(envs.single_action_space.high[0])

        self.actor = Actor(
            self.envs,
            self.device,
            num_struct_elements=self.num_struct_elements,
            att_mask=self.att_mask,
            components_mask=self.components_mask,
        ).to(self.device)
        self.qf1 = SoftQNetwork(
            self.envs,
            self.device,
            num_struct_elements=self.num_struct_elements,
            att_mask=self.att_mask,
            components_mask=self.components_mask,
        ).to(self.device)
        self.qf2 = SoftQNetwork(
            self.envs,
            self.device,
            num_struct_elements=self.num_struct_elements,
            att_mask=self.att_mask,
            components_mask=self.components_mask,
        ).to(self.device)
        self.qf1_target = SoftQNetwork(
            self.envs,
            self.device,
            num_struct_elements=self.num_struct_elements,
            att_mask=self.att_mask,
            components_mask=self.components_mask,
        ).to(self.device)
        self.qf2_target = SoftQNetwork(
            self.envs,
            self.device,
            num_struct_elements=self.num_struct_elements,
            att_mask=self.att_mask,
            components_mask=self.components_mask,
        ).to(self.device)

    def learn(self, weights=None):
        date_time = datetime.now().strftime("%Y.%m.%d_%H-%M")
        run_name = f"{self.args.env_id}__{self.args.exp_name}__{self.args.seed}__{int(time.time())}"

        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )

        # if finetune:
        #     ...

        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.args.q_lr
        )
        actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.args.policy_lr)

        # Automatic entropy tuning
        if self.args.autotune:
            target_entropy = -torch.prod(
                torch.Tensor(self.envs.single_action_space.shape).to(self.device)
            ).item()
            log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            alpha = log_alpha.exp().item()
            a_optimizer = optim.Adam([log_alpha], lr=self.args.q_lr)
        else:
            alpha = self.args.alpha

        self.envs.single_observation_space.dtype = np.float32  # type: ignore
        rb = ReplayBuffer(
            self.args.buffer_size,
            self.envs.single_observation_space,
            self.envs.single_action_space,
            self.device,
            handle_timeout_termination=False,
        )
        start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs, _ = self.envs.reset(seed=self.args.seed)
        for global_step in range(self.args.total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < self.args.learning_starts:
                actions = np.array(
                    [self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)]
                )
            else:
                actions, _, _ = self.actor.get_action(torch.Tensor(obs).to(self.device))
                actions = actions.detach().cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if "final_info" in infos:
                for info in infos["final_info"]:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)  # type: ignore

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > self.args.learning_starts:
                data = rb.sample(self.args.batch_size)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = self.actor.get_action(
                        data.next_observations
                    )
                    qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
                    qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
                    min_qf_next_target = (
                        torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    )
                    next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                    ) * self.args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
                qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if global_step % self.args.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(
                        self.args.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = self.actor.get_action(data.observations)
                        qf1_pi = self.qf1(data.observations, pi)
                        qf2_pi = self.qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        if self.args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = self.actor.get_action(data.observations)
                            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            alpha = log_alpha.exp().item()

                # update the target networks
                if global_step % self.args.target_network_frequency == 0:
                    for param, target_param in zip(
                        self.qf1.parameters(), self.qf1_target.parameters(), strict=False
                    ):
                        target_param.data.copy_(
                            self.args.tau * param.data + (1 - self.args.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.qf2.parameters(), self.qf2_target.parameters(), strict=False
                    ):
                        target_param.data.copy_(
                            self.args.tau * param.data + (1 - self.args.tau) * target_param.data
                        )

                if global_step % 100 == 0:
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/alpha", alpha, global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar(
                        "charts/SPS", int(global_step / (time.time() - start_time)), global_step
                    )
                    if self.args.autotune:
                        writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

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

    def _seed(self) -> None:
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic
