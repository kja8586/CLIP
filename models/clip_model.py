"""MiniCLIP model combining image and text encoders."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder


class MiniCLIP(nn.Module):
    """CLIP-style contrastive model with image and text encoders."""

    def __init__(self):
        super().__init__()
        self.imageEncoder = ImageEncoder()
        self.textEncoder = TextEncoder()

        self.img_proj = nn.Linear(CFG.img_embed_size, CFG.proj_embed_size)
        self.text_proj = nn.Linear(CFG.text_embed_size, CFG.proj_embed_size)

        self.temp = nn.Parameter(torch.tensor(0.07))

    def encode_image(self, x):
        """Encode images to normalized embeddings."""
        img_embeddings = self.imageEncoder(x)
        img_embeddings = self.img_proj(img_embeddings)
        return F.normalize(img_embeddings, p=2, dim=-1)

    def encode_text(self, y):
        """Encode text to normalized embeddings."""
        text_embeddings = self.textEncoder(y)
        text_embeddings = self.text_proj(text_embeddings)
        return F.normalize(text_embeddings, p=2, dim=-1)

    def forward(self, x, y):
        """Compute similarity logits between image and text embeddings."""
        img_embeddings = self.encode_image(x)
        text_embeddings = self.encode_text(y)

        logits = img_embeddings @ text_embeddings.T
        logits = logits / self.temp

        return logits
