"""Training and evaluation for a single fold."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .data import SentenceDataset
from .model import TextClassifier


def _loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    """Return (loss, accuracy, macro_f1, predictions, targets) for a loader."""
    model.eval()
    preds, targets, total_loss = [], [], 0.0
    for batch in loader:
        batch = _move(batch, device)
        logits = model(**batch)
        total_loss += criterion(logits, batch["labels"]).item() * len(batch["labels"])
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        targets.append(batch["labels"].cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    loss = total_loss / len(targets)
    accuracy = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro")
    return loss, accuracy, macro_f1, preds, targets


def train_fold(cfg, tokenizer, train_df, val_df, test_df, device):
    """Train one model and report the test split at the best validation epoch.

    Three splits with distinct jobs: the model fits on `train`, the best epoch
    is chosen on `val`, and the reported numbers (plus the predictions feeding
    the confusion matrix) come from `test`, which is never used for fitting or
    model selection. A full per-epoch history is returned for learning curves.
    """
    def make_loader(df, shuffle):
        ds = SentenceDataset(df["sentence"], df["label"], tokenizer, cfg.max_length)
        return _loader(ds, cfg.batch_size, shuffle, cfg.num_workers)

    train_loader = make_loader(train_df, shuffle=True)
    train_eval_loader = make_loader(train_df, shuffle=False)  # for clean train metrics
    val_loader = make_loader(val_df, shuffle=False)
    test_loader = make_loader(test_df, shuffle=False)

    model = TextClassifier(
        cfg.model_name,
        pooling=cfg.pooling,
        dropout=cfg.dropout,
        freeze_encoder=(cfg.mode == "frozen"),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.warmup_ratio * total_steps), total_steps
    )
    criterion = nn.CrossEntropyLoss()

    history = []
    best = {"val_macro_f1": -1.0}
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = _move(batch, device)
            optimizer.zero_grad()
            logits = model(**batch)
            loss = criterion(logits, batch["labels"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

        tr_loss, tr_acc, tr_f1, _, _ = evaluate(model, train_eval_loader, device, criterion)
        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_loader, device, criterion)
        te_loss, te_acc, te_f1, te_preds, te_targets = evaluate(model, test_loader, device, criterion)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_accuracy": tr_acc, "train_macro_f1": tr_f1,
            "val_loss": va_loss, "val_accuracy": va_acc, "val_macro_f1": va_f1,
            "test_loss": te_loss, "test_accuracy": te_acc, "test_macro_f1": te_f1,
        })

        # epoch selection is on val only; test is snapshotted for reporting
        if va_f1 > best["val_macro_f1"]:
            best = {
                "epoch": epoch,
                "val_macro_f1": va_f1,
                "accuracy": te_acc,
                "macro_f1": te_f1,
                "preds": te_preds,
                "targets": te_targets,
            }

    best["history"] = history
    return best
