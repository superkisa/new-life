# docs and experiment results can be found at
# https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from stable_baselines3.common.buffers import ReplayBuffer
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from transformers import DistilBertConfig, DistilBertModel

from datetime import datetime

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


class CustomModel(nn.Module):
    def __init__(
        self,
        num_struct_elements: int,
        attention_mask: torch.LongTensor,
        components_mask: torch.LongTensor,
        device,
    ):
        self.bert_config = DistilBertConfig(
            vocab_size=10000,
            hidden_size=1,
            num_hidden_layers=2,
            num_attention_heads=1,
            intermediate_size=100,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=50,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            position_embedding_type="absolute",
            use_cache=True,
            classifier_dropout=None,
        )
        super().__init__()
        self.num_struct_elements = num_struct_elements
        self.attention_mask = attention_mask.to(device)
        self.components_mask = components_mask.to(device)
        self.ones_vector = torch.ones(self.num_struct_elements, 1).to(device)
        self.distilbert_1 = DistilBertModel(self.bert_config)
        self.distilbert_2 = DistilBertModel(self.bert_config)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        # num_struct_elements: int = 9,
        # attention_mask: NDArray[np.int_] = None,
        # components_mask: NDArray[np.int_] = None,
    ):
        attention_mask = self.attention_mask
        components_mask = self.components_mask

        embeds = inputs_embeds.repeat(self.num_struct_elements, 1)
        embeds.unsqueeze_(-1)

        outputs_1 = self.distilbert_1(
            # input_ids=torch.ones(input_ids.size()),
            inputs_embeds=embeds,
            attention_mask=attention_mask,
        )

        last_hidden_state_1 = outputs_1["last_hidden_state"]

        input_2 = torch.sum(last_hidden_state_1, dim=2)
        input_2.mul_(components_mask)  # summing through columns
        input_2 = torch.sum(input_2, dim=0)

        input_2 = self.ones_vector @ input_2.view(1, input_2.size()[0])

        input_2 = input_2.view(input_2.size()[0], input_2.size()[1], 1)

        outputs_2 = self.distilbert_2(
            # input_ids=torch.ones(input_2.size()),
            inputs_embeds=input_2,
            attention_mask=attention_mask,
        )
        last_hidden_state_2 = outputs_2["last_hidden_state"]

        input_2 = torch.sum(last_hidden_state_2, dim=2)
        input_2.mul_(components_mask)
        # summing through columns
        input_2 = torch.sum(input_2, dim=0)
        return input_2


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(
        self,
        env,
        device,
        num_struct_elements: int,
        att_mask: torch.LongTensor,
        components_mask: torch.LongTensor,
    ):
        super().__init__()

        self.preprocess_layer = CustomModel(
            num_struct_elements=num_struct_elements,
            attention_mask=att_mask,
            components_mask=components_mask,
            device=device,
        )
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs, action):
        obs = self.preprocess_layer(obs).unsqueeze(0)
        x = torch.cat([obs, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(
        self,
        env,
        device,
        num_struct_elements: int,
        att_mask: torch.LongTensor,
        components_mask: torch.LongTensor,
    ):
        super().__init__()
        self.preprocess_layer = CustomModel(
            num_struct_elements=num_struct_elements,
            attention_mask=att_mask,
            components_mask=components_mask,
            device=device,
        )
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32),
        )

    def forward(self, x):
        x = self.preprocess_layer(x).unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def main(  # noqa: PLR0912, PLR0915
    env,
    n_legs: int,
    num_struct_elements: int,
    att_mask: torch.LongTensor,
    components_mask: torch.LongTensor,
    args: Args,
):
    date_time = datetime.now().strftime("%Y.%m.%d_%H-%M")
    # args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(
        envs,
        device,
        num_struct_elements=num_struct_elements,
        att_mask=att_mask,
        components_mask=components_mask,
    ).to(device)
    qf1 = SoftQNetwork(
        envs,
        device,
        num_struct_elements=num_struct_elements,
        att_mask=att_mask,
        components_mask=components_mask,
    ).to(device)
    qf2 = SoftQNetwork(
        envs,
        device,
        num_struct_elements=num_struct_elements,
        att_mask=att_mask,
        components_mask=components_mask,
    ).to(device)
    qf1_target = SoftQNetwork(
        envs,
        device,
        num_struct_elements=num_struct_elements,
        att_mask=att_mask,
        components_mask=components_mask,
    ).to(device)
    qf2_target = SoftQNetwork(
        envs,
        device,
        num_struct_elements=num_struct_elements,
        att_mask=att_mask,
        components_mask=components_mask,
    ).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32  # type: ignore
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

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
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                )
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                    min_qf_next_target
                ).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters(), strict=False
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters(), strict=False
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
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
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    torch.save(actor.state_dict(), "weights/actor_" + str(n_legs) + "legs_" + date_time + ".pt")
    torch.save(qf1.state_dict(), "weights/qf1_" + str(n_legs) + "legs_" + date_time + ".pt")
    torch.save(qf2.state_dict(), "weights/qf2_" + str(n_legs) + "legs_" + date_time + ".pt")
    torch.save(qf1_target.state_dict(), "weights/qf1_target_" + str(n_legs) + "legs_" + date_time + ".pt")
    torch.save(qf2_target.state_dict(), "weights/qf2_target_" + str(n_legs) + "legs_" + date_time + ".pt")
    envs.close()
    writer.close()


if __name__ == "__main__":
    env = "Ant-v4"
    main(
        env,
        4,
        9,
        torch.from_numpy(np.array([[1] * 6 + [0, 1, 0, 1, 0, 1, 0] + [1] * 7 + [0, 1, 0, 1, 0, 1, 0],
                                   [1] * 7 + [0] * 6 + [1] * 8 + [0] * 6,
                                   [1] * 5 + [0] * 2 + [1] * 2 + [0] * 4 + [1] * 6 + [0] * 2 + [1] * 2 + [0] * 4,
                                   [1] * 5 + [0] * 4 + [1] * 2 + [0] * 2 + [1] * 6 + [0] * 4 + [1] * 2 + [0] * 2,
                                   [1] * 5 + [0] * 6 + [1] * 2 + [0] * 0 + [1] * 6 + [0] * 6 + [1] * 2 + [0] * 0,
                                   [0] * 5 + [1] * 2 + [0] * 12 + [1] * 2 + [0] * 6,
                                   [0] * 7 + [1] * 2 + [0] * 12 + [1] * 2 + [0] * 4,
                                   [0] * 9 + [1] * 2 + [0] * 12 + [1] * 2 + [0] * 2,
                                   [0] * 11 + [1] * 2 + [0] * 12 + [1] * 2 + [0] * 0])).to(torch.int64),  # type: ignore
        torch.from_numpy(np.array([[1] * 5 + [0] * 8 + [1] * 6 + [0] * 8,
                                   [0] * 5 + [1] + [0] * 13 + [1] + [0] * 7,
                                   [0] * 6 + [1] + [0] * 13 + [1] + [0] * 6,
                                   [0] * 7 + [1] + [0] * 13 + [1] + [0] * 5,
                                   [0] * 8 + [1] + [0] * 13 + [1] + [0] * 4,
                                   [0] * 9 + [1] + [0] * 13 + [1] + [0] * 3,
                                   [0] * 10 + [1] + [0] * 13 + [1] + [0] * 2,
                                   [0] * 11 + [1] + [0] * 13 + [1] + [0] * 1,
                                   [0] * 12 + [1] + [0] * 13 + [1] + [0] * 0])).to(torch.int64),  # type: ignore
        Args(env_id=env, learning_starts=500, batch_size=1),
    )
