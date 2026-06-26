"""Experiment configuration.

A run is defined by three choices: the backbone (`model_name`), the head that
turns token embeddings into a prediction (`head`), and whether the encoder is
fine-tuned (`mode`). The optimisation defaults follow from head and mode, so the
scripts stay free of magic numbers.
"""

from dataclasses import dataclass, replace


# Backbones we compare. Kept in one place so adding or dropping one is a
# single-line edit rather than a hunt through the notebook.
MODELS = [
    "bert-base-uncased",
    "roberta-base",
    "distilbert-base-uncased",
    "bert-base-multilingual-cased",
    "sentence-transformers/LaBSE",
]

# Heads: mean / cls feed a linear classifier on a pooled vector; lstm is the
# Hollenstein et al. (2021) text component (BiLSTM over the full token sequence).
HEADS = ["mean", "cls", "lstm"]
MODES = ["frozen", "finetune"]

# Datasets share the same schema (a `sentence` column + a -1/0/1 `sentiment_label`),
# so a dataset is just a different csv. TeCo is the Persian counterpart to ZuCo.
DATASETS = {
    "zuco": "data/zuco_sentiment_labels_task1_fixed.csv",
    "teco": "data/teco_sentiment_labels_task1.csv",
}


def optim_defaults(head, mode):
    """Learning rates, epochs and batch size for a given head and mode.

    Frozen runs only train the head, so they can afford a higher learning rate
    and need more epochs to converge. Fine-tuning uses a small encoder rate; the
    lstm head still wants a larger rate on its own (randomly initialised) weights.
    """
    if mode == "frozen":
        return dict(encoder_lr=0.0, head_lr=1e-3, weight_decay=0.0,
                    epochs=30 if head == "lstm" else 20, batch_size=32)
    return dict(encoder_lr=2e-5, head_lr=1e-3 if head == "lstm" else 2e-5,
                weight_decay=0.01, epochs=10, batch_size=16)


@dataclass
class Config:
    # data
    dataset: str = "zuco"          # "zuco" | "teco"; selects the csv via DATASETS
    csv_path: str = "data/zuco_sentiment_labels_task1_fixed.csv"
    max_length: int = 64           # longest sentence is ~43 words; 64 tokens is safe

    # model
    model_name: str = "bert-base-uncased"
    head: str = "mean"             # "mean" | "cls" | "lstm"
    mode: str = "frozen"           # "frozen" | "finetune"
    dropout: float = 0.1           # linear-head dropout

    # lstm head (Hollenstein et al. 2021 text component)
    lstm_dim: int = 256
    dense_dim: int = 128
    lstm_dropout: float = 0.3

    # optimisation (filled in by with_setup)
    encoder_lr: float = 0.0
    head_lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 32
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # evaluation
    n_folds: int = 5               # outer folds; each held-out fold is the test set
    val_size: float = 0.15         # inner split of each fold's train portion, for epoch selection
    seed: int = 42

    # misc
    output_dir: str = "results"
    num_workers: int = 2

    def with_setup(self, head, mode):
        """Return a copy configured for one (head, mode) with its optimiser defaults."""
        return replace(self, head=head, mode=mode, **optim_defaults(head, mode))

    @property
    def head_dropout(self):
        return self.lstm_dropout if self.head == "lstm" else self.dropout


# Label space. ZuCo task 1 uses -1/0/1; we remap to contiguous ids for the loss
# and keep the inverse around for readable reports.
LABEL_TO_ID = {-1: 0, 0: 1, 1: 2}
ID_TO_NAME = {0: "negative", 1: "neutral", 2: "positive"}
NUM_CLASSES = len(LABEL_TO_ID)
