from typing import TYPE_CHECKING

import numpy as np
import torch
from jaxtyping import Float, Int64
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import DistilBertConfig, DistilBertModel

if TYPE_CHECKING:
    from transformers.modeling_outputs import BaseModelOutput


class BertsAttention(nn.Module):
    def __init__(
        self,
        num_struct_elements: int,
        attention_mask: Int64[torch.Tensor, "..."],
        components_mask: Int64[torch.Tensor, "..."],
    ):
        super().__init__()

        self.num_struct_elements = num_struct_elements
        self.attention_mask = attention_mask
        self.components_mask = components_mask
        self.ones_vector = torch.ones(self.num_struct_elements, 1)

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
        self.distilbert_1 = DistilBertModel(self.bert_config)
        self.distilbert_2 = DistilBertModel(self.bert_config)

    def __call__(self, inputs_embeds: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        return super().__call__(inputs_embeds)

    def forward(self, inputs_embeds: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        attention_mask = self.attention_mask
        components_mask = self.components_mask

        embeds = inputs_embeds.repeat(self.num_struct_elements, 1)
        embeds.unsqueeze_(-1)

        outputs_1: BaseModelOutput = self.distilbert_1(
            attention_mask=attention_mask,
            inputs_embeds=embeds,
        )

        last_hidden_state_1 = outputs_1["last_hidden_state"]

        input_2 = torch.sum(last_hidden_state_1, dim=2)
        input_2.mul_(components_mask)  # summing through columns
        input_2 = torch.sum(input_2, dim=0)

        input_2 = self.ones_vector @ input_2.view(1, input_2.size()[0])

        input_2 = input_2.view(input_2.size()[0], input_2.size()[1], 1)

        outputs_2: BaseModelOutput = self.distilbert_2(
            attention_mask=attention_mask,
            inputs_embeds=input_2,
        )
        last_hidden_state_2 = outputs_2["last_hidden_state"]

        input_2 = torch.sum(last_hidden_state_2, dim=2)
        input_2.mul_(components_mask)

        input_2 = torch.sum(input_2, dim=0)  # summing through columns
        return input_2


class AttentionLayer(nn.Module):
    def __init__(
        self, mask: Float[Tensor, "batch num_obs num_obs"], hidden_dim: int, *, bias: bool = True
    ):
        """Constructs a new RL-oriented attention layer.

        Args:
            mask: attention mask of shape `batch num_obs num_obs`
            hidden_dim: dimension of QKV weights
        """
        super().__init__()
        self.register_buffer("mask", mask)
        weights_shape = (1, 1, hidden_dim)

        self.W_q = nn.Parameter(torch.randn(weights_shape))
        self.W_k = nn.Parameter(torch.randn(weights_shape))
        self.W_v = nn.Parameter(torch.randn(weights_shape))

        self.b_q = nn.Parameter(torch.randn(weights_shape)) if bias else 0.0
        self.b_k = nn.Parameter(torch.randn(weights_shape)) if bias else 0.0
        self.b_v = nn.Parameter(torch.randn(weights_shape)) if bias else 0.0

        self.softmax = nn.Softmax(dim=1)

    def forward(self, X: Float[Tensor, "batch num_obs"]) -> Float[Tensor, "batch num_obs"]:
        X = X.unsqueeze(-1)
        Q = X @ self.W_q + self.b_q
        K = X @ self.W_k + self.b_k
        V = X @ self.W_v + self.b_v

        scores = Q @ K.mT
        masked_scores = scores * self.mask
        soft_scores = self.softmax(masked_scores)
        soft_scores_value = soft_scores @ V
        Z = torch.sum(soft_scores_value, dim=2)
        return Z


class SoftQNetwork(nn.Module):
    """Custom Critic network.

    Attributes:
        preprocess_layer: attention layer, ℝ `batch × num_obs` -> ℝ `batch × num_obs`
        fc: fully-connected layer ℝ `batch × num_act+num_obs` -> `batch 1`

    """

    def __init__(
        self,
        att_mask: Int64[torch.Tensor, "..."],
        num_obs: int,
        num_act: int,
        fc_hidden_dim: int = 256,
        num_attention_layers: int = 3,
    ):
        super().__init__()

        self.preprocess_layer = nn.Sequential(
            *[AttentionLayer(mask=att_mask, hidden_dim=16) for _ in range(num_attention_layers)]
        )

        self.fc = nn.Sequential(
            nn.Linear(num_obs + num_act, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, 1),
        )

    def forward(
        self, obs: Float[Tensor, "batch num_obs"], action: Float[Tensor, "batch num_act"]
    ) -> Float[Tensor, "batch 1"]:
        # obs = self.preprocess_layer(obs)
        flat_obs_act = torch.cat([obs, action], dim=1)
        out = self.fc(flat_obs_act)
        return out


class Actor(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    def __init__(
        self,
        att_mask: Int64[torch.Tensor, "..."],
        num_obs: int,
        num_act: int,
        fc_hidden_dim: int = 256,
        num_attention_layers: int = 3,
    ):
        super().__init__()

        self.preprocess_layer = nn.Sequential(
            *[AttentionLayer(mask=att_mask, hidden_dim=16) for _ in range(num_attention_layers)]
        )

        self.fc = nn.Sequential(
            nn.Linear(np.array(num_obs).prod(), fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, fc_hidden_dim),
            nn.ReLU(),
        )

        # self.fc1 = nn.Linear(np.array(num_obs).prod(), fc_hidden_dim)
        # self.fc2 = nn.Linear(fc_hidden_dim, fc_hidden_dim)
        self.fc_mean = nn.Linear(fc_hidden_dim, num_act)
        self.fc_logstd = nn.Linear(fc_hidden_dim, num_act)

    def forward(self, x):
        # x = self.preprocess_layer(x)
        x = self.fc(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean
    
    def get_action0(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + 1e-6)
        # log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean


# class CleanRLActor(nn.Module):
#     LOG_STD_MAX = 2
#     LOG_STD_MIN = -5

#     def __init__(
#         self,
#         env,
#         att_mask: Int64[torch.Tensor, "..."],
#         num_attention_layers: int = 3,
#     ):
#         super().__init__()
#         self.preprocess_layer = nn.Sequential(
#             *[AttentionLayer(mask=att_mask, hidden_dim=16) for _ in range(num_attention_layers)]
#         )
#         self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
#         self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
#         # action rescaling
#         self.register_buffer(
#             "action_scale",
#             torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32),
#         )
#         self.register_buffer(
#             "action_bias",
#             torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32),
#         )

#     def forward(self, x):
#         x = self.preprocess_layer(x).squeeze(-1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         mean = self.fc_mean(x)
#         log_std = self.fc_logstd(x)
#         log_std = torch.tanh(log_std)
#         log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)

#         return mean, log_std

#     def get_action(self, x):
#         mean, log_std = self(x)
#         std = log_std.exp()
#         normal = torch.distributions.Normal(mean, std)
#         x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
#         y_t = torch.tanh(x_t)
#         action = y_t * self.action_scale + self.action_bias
#         log_prob = normal.log_prob(x_t)
#         # Enforcing Action Bound
#         log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
#         log_prob = log_prob.sum(1, keepdim=True)
#         mean = torch.tanh(mean) * self.action_scale + self.action_bias
#         return action, log_prob, mean
