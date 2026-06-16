"""Dataset loading, tokenisation and cross-validation splits."""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from .config import LABEL_TO_ID


def load_dataframe(csv_path):
    """Read the ZuCo csv and return sentences with contiguous integer labels."""
    df = pd.read_csv(csv_path)
    df = df[["sentence", "sentiment_label"]].dropna()
    df["sentence"] = df["sentence"].str.strip()
    df = df[df["sentence"].str.len() > 0]
    df["label"] = df["sentiment_label"].astype(int).map(LABEL_TO_ID)
    return df.reset_index(drop=True)


class SentenceDataset(Dataset):
    """Holds pre-tokenised sentences for a single fold.

    Tokenising up front keeps the training loop simple and means the slow part
    only happens once per fold rather than every epoch.
    """

    def __init__(self, sentences, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            list(sentences),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def fold_indices(labels, n_folds, seed):
    """Yield (train_idx, val_idx) pairs for stratified k-fold.

    Stratifying matters here: the dataset is small (~400 rows) and the three
    sentiment classes are only roughly balanced, so a plain split would leave
    the per-fold class ratios drifting.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    labels = np.asarray(labels)
    yield from skf.split(np.zeros_like(labels), labels)
