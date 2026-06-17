"""The classifier: a transformer encoder with a switchable prediction head."""

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
    """Encoder plus one of three heads.

    mean / cls
        Pool the token embeddings into one vector (masked mean, or the [CLS]
        position) and classify it with a single linear layer.
    lstm
        The text component from Hollenstein et al. (2021): a bidirectional LSTM
        over the full token sequence, flattened, then a dense + dropout + output
        layer. Reproduces their architecture on top of a (here, frozen) encoder.

    `freeze_encoder` decides whether the encoder is fine-tuned; the head is the
    same regardless, so frozen and fine-tuned runs differ only by that flag.
    """

    def __init__(self, model_name, head="mean", dropout=0.1, freeze_encoder=False,
                 max_length=64, lstm_dim=256, dense_dim=128):
        super().__init__()
        self.head = head
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        if head == "lstm":
            self.lstm = nn.LSTM(hidden_size, lstm_dim, batch_first=True, bidirectional=True)
            self.dense = nn.Linear(max_length * 2 * lstm_dim, dense_dim)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(dense_dim, NUM_CLASSES)
        else:
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

    def forward(self, input_ids, attention_mask, **_):
        sequence = self.encoder(input_ids=input_ids,
                                attention_mask=attention_mask).last_hidden_state

        if self.head == "lstm":
            # BiLSTM over every token, then flatten the whole sequence like the
            # Keras Flatten() in the reference implementation.
            lstm_out, _ = self.lstm(sequence)
            flat = lstm_out.flatten(start_dim=1)
            hidden = torch.relu(self.dense(flat))
            return self.classifier(self.dropout(hidden))

        pooled = sequence[:, 0] if self.head == "cls" else masked_mean(sequence, attention_mask)
        return self.classifier(self.dropout(pooled))

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)
