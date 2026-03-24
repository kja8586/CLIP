"""Image encoder using timm backbone."""

import timm
import torch.nn as nn

from config import CFG


class ImageEncoder(nn.Module):
    """Image encoder using a timm pretrained model."""

    def __init__(self, model_name=None, pre_trained=None, trainable=None):
        super().__init__()
        model_name = model_name or CFG.imageModel
        pre_trained = pre_trained if pre_trained is not None else CFG.pre_trained
        trainable = trainable if trainable is not None else CFG.trainable

        self.model = timm.create_model(
            model_name, pretrained=pre_trained, num_classes=0, global_pool="avg"
        )

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
