"""Cross-validation runner that ties the pieces together."""

import json
import os
import random

import numpy as np
import torch
from transformers import AutoTokenizer

from .data import fold_indices, load_dataframe
from .engine import train_fold


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cross_validate(cfg, verbose=True):
    """Run stratified k-fold CV and return a summary of per-fold scores.

    A fresh model is trained from scratch on every fold; we report the best
    validation epoch per fold and then average across folds.
    """
    set_seed(cfg.seed)
    device = pick_device()

    df = load_dataframe(cfg.csv_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    fold_scores = []
    splits = fold_indices(df["label"].values, cfg.n_folds, cfg.seed)
    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        result = train_fold(
            cfg, tokenizer, df.iloc[train_idx], df.iloc[val_idx], device
        )
        fold_scores.append(result)
        if verbose:
            print(
                f"  fold {fold}/{cfg.n_folds}  "
                f"acc {result['accuracy']:.3f}  "
                f"macro-f1 {result['macro_f1']:.3f}  "
                f"(best epoch {result['epoch']})"
            )

    summary = _summarise(cfg, fold_scores)
    if verbose:
        print(
            f"  => {cfg.mode:<8} {cfg.model_name}  "
            f"acc {summary['accuracy_mean']:.3f} +/- {summary['accuracy_std']:.3f}  "
            f"macro-f1 {summary['macro_f1_mean']:.3f} +/- {summary['macro_f1_std']:.3f}"
        )
    return summary


def _summarise(cfg, fold_scores):
    acc = [s["accuracy"] for s in fold_scores]
    f1 = [s["macro_f1"] for s in fold_scores]
    return {
        "model_name": cfg.model_name,
        "mode": cfg.mode,
        "n_folds": cfg.n_folds,
        "accuracy_mean": float(np.mean(acc)),
        "accuracy_std": float(np.std(acc)),
        "macro_f1_mean": float(np.mean(f1)),
        "macro_f1_std": float(np.std(f1)),
        "folds": fold_scores,
    }


def save_summary(summary, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    name = f"{summary['mode']}_{summary['model_name'].replace('/', '-')}.json"
    path = os.path.join(output_dir, name)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path
