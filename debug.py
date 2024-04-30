import torch
from torch import nn
from transformers import DistilBertConfig, DistilBertModel


class CustomModel(nn.Module):
    def __init__(self):
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
        self.distilbert_1 = DistilBertModel(self.bert_config)
        self.distilbert_2 = DistilBertModel(self.bert_config)

    def forward(self, input_ids, num_struct_elements=9, attention_mask=None, components_mask=None):
        input_ids = input_ids.view(input_ids.size()[0], input_ids.size()[1], 1)
        outputs_1 = self.distilbert_1(
            # input_ids=torch.ones(input_ids.size()),
            inputs_embeds=input_ids,
            attention_mask=attention_mask,
        )

        last_hidden_state_1 = outputs_1["last_hidden_state"]

        input_2 = torch.sum(last_hidden_state_1, axis=2)
        input_2.mul_(components_mask)  # summing through columns
        input_2 = torch.sum(input_2, axis=0)
        ones_vector = torch.ones(num_struct_elements, 1)
        input_2 = ones_vector @ input_2.view(1, input_2.size()[0])

        input_2 = input_2.view(input_2.size()[0], input_2.size()[1], 1)

        outputs_2 = self.distilbert_2(
            # input_ids=torch.ones(input_2.size()),
            inputs_embeds=input_2,
            attention_mask=attention_mask,
        )
        last_hidden_state_2 = outputs_2["last_hidden_state"]

        input_2 = torch.sum(last_hidden_state_2, axis=2)
        input_2.mul_(components_mask)
        # summing through columns
        input_2 = torch.sum(input_2, axis=0)
        return input_2


v = CustomModel()

out = v(
    input_ids=torch.tensor(
        [[0.9086, 0.0, 0.4586, 0.50, 0.1300], [0.190, 0.1, 0.80, 0.50, 0.1300]]
    ),
    num_struct_elements=2,
    attention_mask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
    components_mask=torch.tensor([[1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]),
)
