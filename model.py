from transformers import ElectraModel, ElectraTokenizerFast
from torch.utils.data import DataLoader
from typing import List, Dict
import torch.nn as nn
import torch


class SentenceBert(nn.Module):

    def __init__(self, model=None, pooling_type: str = "mean") -> None:
        super().__init__()
        name = "monologg/koelectra-base-v3-discriminator"
        self.model = model if model else ElectraModel.from_pretrained(name)

        if pooling_type in ["mean", "max", "cls"] and type(pooling_type) == str:
            self.pooling_type = pooling_type
        else:
            raise ValueError("'pooling_type' only ['mean','max','cls'] possible")

    def forward(self, **kwargs):
        attention_mask = kwargs["attention_mask"]
        last_hidden_state = self.model(**kwargs)["last_hidden_state"]

        if self.pooling_type == "cls":
            result = last_hidden_state[:, 0]

        if self.pooling_type == "max":
            num_of_tokens = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            last_hidden_state[num_of_tokens == 0] = -1e9
            result = torch.max(last_hidden_state, 1)[0]

        if self.pooling_type == "mean":
            num_of_tokens = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

            sum_embeddings = torch.sum(last_hidden_state * num_of_tokens, 1)

            total_num_of_tokens = num_of_tokens.sum(1)
            total_num_of_tokens = torch.clamp(total_num_of_tokens, min=1e-9)

            result = sum_embeddings / total_num_of_tokens

        return {"sentence_embedding": result}