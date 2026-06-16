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
def evaluate(model, loader, device):
    """Return (accuracy, macro_f1, predictions, targets) for a data loader."""
    model.eval()
    preds, targets = [], []
    for batch in loader:
        batch = _move(batch, device)
        logits = model(**batch)
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        targets.append(batch["labels"].cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    accuracy = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro")
    return accuracy, macro_f1, preds, targets


def train_fold(cfg, tokenizer, train_df, val_df, device):
    """Train one model on a single train/val split and report its best epoch."""
    train_ds = SentenceDataset(train_df["sentence"], train_df["label"], tokenizer, cfg.max_length)
    val_ds = SentenceDataset(val_df["sentence"], val_df["label"], tokenizer, cfg.max_length)

    train_loader = _loader(train_ds, cfg.batch_size, True, cfg.num_workers)
    val_loader = _loader(val_ds, cfg.batch_size, False, cfg.num_workers)

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

    best = {"accuracy": 0.0, "macro_f1": 0.0, "epoch": 0, "preds": None, "targets": None}
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

        accuracy, macro_f1, preds, targets = evaluate(model, val_loader, device)
        if macro_f1 > best["macro_f1"]:
            best = {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "epoch": epoch,
                "preds": preds,
                "targets": targets,
            }

    return best
