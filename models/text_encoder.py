"""Text encoder using DistilBERT."""

import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig

from config import CFG


class TextEncoder(nn.Module):
    """Text encoder using DistilBERT with mean pooling."""

    def __init__(self, model_name=None, pre_trained=None, trainable=None):
        super().__init__()
        model_name = model_name or CFG.textModel
        pre_trained = pre_trained if pre_trained is not None else CFG.pre_trained
        trainable = trainable if trainable is not None else CFG.trainable

        if pre_trained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        out = self.model(input_ids=x["input_ids"], attention_mask=x["attention_mask"])
        mask = x["attention_mask"].unsqueeze(-1)  # (B, seq_len, 1)
        return (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
