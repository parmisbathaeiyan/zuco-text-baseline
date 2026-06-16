"""The classifier: a transformer encoder with a linear sentiment head."""

import torch
import torch.nn as nn
from transformers import AutoModel

from .config import NUM_CLASSES


def masked_mean(hidden_state, attention_mask):
    """Average token embeddings, ignoring padding."""
    mask = attention_mask.unsqueeze(-1).type_as(hidden_state)
    summed = (hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class TextClassifier(nn.Module):
    """Encoder + dropout + linear layer.

    `freeze_encoder=True` gives the frozen baseline (a linear probe on top of
    fixed features); `False` fine-tunes the whole stack. Both share this one
    class so the two setups differ only by a flag, never by code path.
    """

    def __init__(self, model_name, pooling="mean", dropout=0.1, freeze_encoder=False):
        super().__init__()
        self.pooling = pooling
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, NUM_CLASSES)

        self.frozen = freeze_encoder
        if freeze_encoder:
            self.encoder.requires_grad_(False)

    def train(self, mode=True):
        """Switch to train mode, but never wake a frozen encoder's dropout/LN."""
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def _pool(self, outputs, attention_mask):
        if self.pooling == "cls":
            return outputs.last_hidden_state[:, 0]
        return masked_mean(outputs.last_hidden_state, attention_mask)

    def forward(self, input_ids, attention_mask, **_):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs, attention_mask)
        return self.classifier(self.dropout(pooled))

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)
