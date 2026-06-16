"""Experiment configuration.

Everything that controls a run lives here so the scripts stay free of magic
numbers. `run.py` overrides any of these fields from the command line.
"""

from dataclasses import dataclass, field, asdict


# The frozen probe and the fine-tuned model want very different optimisation
# regimes, so we keep a small default block for each and pick one at run time.
FROZEN_DEFAULTS = dict(lr=1e-3, epochs=20, batch_size=32, weight_decay=0.0)
FINETUNE_DEFAULTS = dict(lr=2e-5, epochs=4, batch_size=16, weight_decay=0.01)


@dataclass
class Config:
    # data
    csv_path: str = "data/zuco_sentiment_labels_task1_fixed.csv"
    max_length: int = 64           # longest sentence is ~43 words, 64 tokens is safe

    # model
    model_name: str = "bert-base-uncased"
    pooling: str = "mean"          # "mean" (masked average) or "cls"
    dropout: float = 0.1

    # training
    mode: str = "finetune"         # "frozen" or "finetune"
    lr: float = 2e-5
    epochs: int = 4
    batch_size: int = 16
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # evaluation
    n_folds: int = 5
    seed: int = 42

    # misc
    output_dir: str = "results"
    num_workers: int = 2

    def with_mode(self, mode: str) -> "Config":
        """Return a copy with the optimisation defaults for `mode` applied."""
        defaults = FROZEN_DEFAULTS if mode == "frozen" else FINETUNE_DEFAULTS
        return Config(**{**asdict(self), "mode": mode, **defaults})


# Label space. ZuCo task 1 uses -1/0/1; we remap to contiguous ids for the loss
# and keep the inverse around for readable reports.
LABEL_TO_ID = {-1: 0, 0: 1, 1: 2}
ID_TO_NAME = {0: "negative", 1: "neutral", 2: "positive"}
NUM_CLASSES = len(LABEL_TO_ID)
