# docs and experiment results can be found at
# https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from stable_baselines3.common.buffers import ReplayBuffer
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from transformers import DistilBertConfig, DistilBertModel


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


def main(
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

class SAC:
    def __init__(self, ):
        ...