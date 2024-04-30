import torch
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from transformers import DistilBertConfig, DistilBertModel
from transformers.modeling_outputs import BaseModelOutput


class CustomModel(nn.Module):
    def __init__(self):
        self.bert_config = DistilBertConfig(
            vocab_size=10000,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
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
        self.distilbert_1 = DistilBertModel(self.bert_config)
        self.distilbert_2 = DistilBertModel(self.bert_config)

    def forward(
        self,
        input_ids: Float[Tensor, "dim1"],
        attention_mask,
        components_mask,
        num_struct_elements=9,
    ):
        input_data = input_ids.expand(num_struct_elements, -1)       
        outputs_1: BaseModelOutput = self.distilbert_1(
            input_data,
            attention_mask=attention_mask,
        )

        last_hidden_state_1 = outputs_1.last_hidden_state

        input_2 = torch.sum(last_hidden_state_1, dim=2)
        input_2.mul_(components_mask)  # summing through columns
        input_2 = torch.sum(input_2, dim=0)
        ones_vector = torch.ones(num_struct_elements, 1)
        input_2 = ones_vector @ input_2.view(1, input_2.size()[0])
        outputs_2: BaseModelOutput = self.distilbert_2(
            input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state_2 = outputs_2.last_hidden_state

        input_2 = torch.sum(last_hidden_state_2, dim=2)
        input_2.mul_(components_mask)
        # summing through columns
        input_2 = torch.sum(input_2, dim=0)
        return input_2


class QHead(nn.Module):
    def __init__(self, size_input, size_action):
        super().__init__()
        self.size_input = size_input
        self.size_action = size_action
        self.layer = nn.Sequential(
            nn.Linear(self.size_input, self.size_action),
            nn.LeakyReLU(0.01),
            nn.Linear(self.size_action, 1),
        )

    def forward(self, input):
        return self.layer(input)


class PiHead(nn.Module):
    def __init__(self, size_input, size_action):
        super().__init__()
        self.size_input = size_input
        self.size_action = size_action
        self.layer = nn.Sequential(
            nn.Linear(self.size_input, self.size_action),
            nn.LeakyReLU(0.01),
        )

        self.last_lin = nn.Linear(1, 2)

    def forward(self, input):
        output_1 = self.layer(input)
        # ones_vector = torch.ones(self.size_action, 1)
        # output_1 = ones_vector @ output_1.view(1, self.size_action)
        return self.last_lin(output_1.view(self.size_action, 1))
