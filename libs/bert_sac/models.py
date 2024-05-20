import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from transformers import DistilBertConfig, DistilBertModel


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


class Actor(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

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
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (
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
