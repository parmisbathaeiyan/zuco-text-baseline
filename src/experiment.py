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
from .plots import save_confusion_matrix


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
        "val_size": cfg.val_size,
        "accuracy_mean": float(np.mean(acc)),
        "accuracy_std": float(np.std(acc)),
        "macro_f1_mean": float(np.mean(f1)),
        "macro_f1_std": float(np.std(f1)),
        "folds": fold_scores,
    }


def save_summary(summary, output_dir):
    """Write the JSON summary and a confusion-matrix PNG next to it."""
    os.makedirs(output_dir, exist_ok=True)
    stem = f"{summary['mode']}_{summary['model_name'].replace('/', '-')}"

    json_path = os.path.join(output_dir, stem + ".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    png_path = os.path.join(output_dir, stem + "_confusion.png")
    save_confusion_matrix(
        np.array(summary["confusion_matrix"]),
        summary["class_names"],
        png_path,
        title=f"{summary['mode']} - {summary['model_name']}",
    )
    return json_path
