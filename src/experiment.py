"""Cross-validation runner that ties the pieces together."""

import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from .config import ID_TO_NAME
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
    """Run nested stratified CV and return a summary of per-fold test scores.

    The outer k-fold defines the test set for each fold; a stratified slice of
    the remaining data becomes the validation set used to pick the best epoch.
    A fresh model is trained per fold and the reported numbers are on the test
    splits, pooled across folds for the confusion matrix.
    """
    set_seed(cfg.seed)
    device = pick_device()

    df = load_dataframe(cfg.csv_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    fold_scores = []
    oof_preds, oof_targets = [], []  # out-of-fold test predictions, pooled for the matrix
    splits = fold_indices(df["label"].values, cfg.n_folds, cfg.seed)
    for fold, (fit_idx, test_idx) in enumerate(splits, start=1):
        fit_df, test_df = df.iloc[fit_idx], df.iloc[test_idx]
        train_df, val_df = train_test_split(
            fit_df,
            test_size=cfg.val_size,
            stratify=fit_df["label"],
            random_state=cfg.seed,
        )

        result = train_fold(cfg, tokenizer, train_df, val_df, test_df, device)
        oof_preds.append(result.pop("preds"))
        oof_targets.append(result.pop("targets"))
        fold_scores.append(result)
        if verbose:
            print(
                f"  fold {fold}/{cfg.n_folds}  "
                f"test acc {result['accuracy']:.3f}  "
                f"test macro-f1 {result['macro_f1']:.3f}  "
                f"(best epoch {result['epoch']})"
            )

    class_ids = sorted(ID_TO_NAME)
    cm = confusion_matrix(
        np.concatenate(oof_targets), np.concatenate(oof_preds), labels=class_ids
    )

    summary = _summarise(cfg, fold_scores)
    summary["class_names"] = [ID_TO_NAME[i] for i in class_ids]
    summary["confusion_matrix"] = cm.tolist()  # rows = true, cols = predicted
    if verbose:
        print(
            f"  => {cfg.head}/{cfg.mode}  {cfg.model_name}  "
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
        "head": cfg.head,
        "n_folds": cfg.n_folds,
        "val_size": cfg.val_size,
        "accuracy_mean": float(np.mean(acc)),
        "accuracy_std": float(np.std(acc)),
        "macro_f1_mean": float(np.mean(f1)),
        "macro_f1_std": float(np.std(f1)),
        "folds": fold_scores,
    }


def setup_dir(output_dir, head, mode):
    """The distinctively named folder that holds one setup's per-model results."""
    return os.path.join(output_dir, f"{head}_{mode}")


def result_path(output_dir, head, mode, model_name):
    """Where a single (model, head, mode) summary lives."""
    return os.path.join(setup_dir(output_dir, head, mode),
                        model_name.replace("/", "-") + ".json")


def save_summary(summary, output_dir):
    """Write one JSON summary into its setup folder. Plots are built separately."""
    path = result_path(output_dir, summary["head"], summary["mode"], summary["model_name"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path
