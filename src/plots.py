"""Plotting helpers. Kept separate so the training code has no matplotlib pull.

Everything here is driven by the saved JSON summaries, so plots can be rebuilt
at any time without rerunning a single model.
"""

import matplotlib
matplotlib.use("Agg")  # write files without needing a display
import matplotlib.pyplot as plt
import numpy as np

from .config import HEADS

_METRICS = [("loss", "loss"), ("accuracy", "accuracy"), ("macro_f1", "macro-F1")]


def short_name(model_name):
    """Drop the org prefix so labels stay readable (sentence-transformers/...)."""
    return model_name.split("/")[-1]


def head_of(summary):
    """Head used for a run, falling back to the older `pooling` field."""
    return summary.get("head", summary.get("pooling", "mean"))


def _curve(history, key):
    """Mean and std of a per-epoch metric across folds (folds x epochs)."""
    stacked = np.array([[h[key] for h in fold] for fold in history])
    return stacked.mean(axis=0), stacked.std(axis=0)


# --- confusion matrices ------------------------------------------------------

def _draw_cm(ax, cm, class_names, title):
    row_totals = cm.sum(axis=1, keepdims=True).clip(min=1)
    rates = cm / row_totals
    ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    short = [c[:3] for c in class_names]
    ax.set_xticks(range(len(short)), short)
    ax.set_yticks(range(len(short)), short)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    for i, j in np.ndindex(cm.shape):
        colour = "white" if rates[i, j] > 0.5 else "black"
        ax.text(j, i, cm[i, j], ha="center", va="center", color=colour, fontsize=8)


def plot_confusion_grid(summaries, path, title):
    """One confusion matrix per backbone, side by side, for a single setup."""
    summaries = [s for s in summaries if s.get("confusion_matrix")]
    summaries = sorted(summaries, key=lambda s: short_name(s["model_name"]))
    if not summaries:
        return
    n = len(summaries)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.3), squeeze=False)
    for ax, s in zip(axes[0], summaries):
        _draw_cm(ax, np.array(s["confusion_matrix"]), s["class_names"],
                 short_name(s["model_name"]))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- score bars and overview -------------------------------------------------

def plot_grouped_scores(summaries, bar_key, path, title):
    """Grouped bars of test accuracy and macro-F1.

    x axis is the backbone; the bars within each group are the distinct values
    of `bar_key` ("head" or "mode").
    """
    keyfn = head_of if bar_key == "head" else (lambda s: s[bar_key])
    models = sorted({s["model_name"] for s in summaries}, key=short_name)
    bars = sorted({keyfn(s) for s in summaries})
    lookup = {(s["model_name"], keyfn(s)): s for s in summaries}

    x = np.arange(len(models))
    width = 0.8 / max(len(bars), 1)

    fig, axes = plt.subplots(1, 2, figsize=(max(8, 1.7 * len(models)), 4.5))
    for ax, metric, label in [(axes[0], "accuracy", "accuracy"),
                              (axes[1], "macro_f1", "macro-F1")]:
        for k, b in enumerate(bars):
            means = [lookup[(m, b)][f"{metric}_mean"] if (m, b) in lookup else 0 for m in models]
            errs = [lookup[(m, b)][f"{metric}_std"] if (m, b) in lookup else 0 for m in models]
            ax.bar(x + (k - (len(bars) - 1) / 2) * width, means, width,
                   yerr=errs, capsize=3, label=str(b))
        ax.set_xticks(x, [short_name(m) for m in models], rotation=30, ha="right")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(frameon=False, title=bar_key)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_overview_heatmap(summaries, metric, path):
    """Heatmap of one metric: rows are head/mode setups, columns are backbones.

    Rows are ordered fine-tuned on top, frozen on the bottom, with the heads in
    their canonical order within each block and a divider between the two.
    """
    models = sorted({s["model_name"] for s in summaries}, key=short_name)
    mode_rank = {"finetune": 0, "frozen": 1}
    head_rank = {h: i for i, h in enumerate(HEADS)}
    setups = sorted(
        {(head_of(s), s["mode"]) for s in summaries},
        key=lambda hm: (mode_rank.get(hm[1], 9), head_rank.get(hm[0], 9)),
    )
    lookup = {(s["model_name"], head_of(s), s["mode"]): s for s in summaries}

    grid = np.full((len(setups), len(models)), np.nan)
    for i, (head, mode) in enumerate(setups):
        for j, m in enumerate(models):
            s = lookup.get((m, head, mode))
            if s:
                grid[i, j] = s[f"{metric}_mean"]

    fig, ax = plt.subplots(figsize=(1.15 * len(models) + 1.8, 0.78 * len(setups) + 1.2))
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(models)), [short_name(m) for m in models],
                  rotation=30, ha="right", fontsize=13)
    ax.set_yticks(range(len(setups)), [f"{mode} · {head}" for head, mode in setups], fontsize=13)
    for i in range(len(setups)):
        for j in range(len(models)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        color="white" if grid[i, j] < 0.6 else "black",
                        fontsize=15, fontweight="medium")

    # white line between the fine-tuned block (top) and the frozen block
    split = sum(1 for _, mode in setups if mode_rank.get(mode, 9) == 0)
    if 0 < split < len(setups):
        ax.axhline(split - 0.5, color="white", linewidth=2.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=11)
    ax.set_title(f"{metric.replace('_', '-')} (mean over folds)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- learning curves (aggregated over folds, one line per backbone) ----------

def plot_model_curves(summaries, path, title, split="test"):
    """For a single setup, overlay every backbone's loss / accuracy / macro-F1."""
    # legacy results may predate per-epoch history; skip those rather than crash
    summaries = [s for s in summaries
                 if s.get("folds") and "history" in s["folds"][0]]
    if not summaries:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (metric, label) in zip(axes, _METRICS):
        for s in summaries:
            histories = [fold["history"] for fold in s["folds"]]
            epochs = [h["epoch"] for h in histories[0]]
            mean, std = _curve(histories, f"{split}_{metric}")
            line, = ax.plot(epochs, mean, label=short_name(s["model_name"]))
            ax.fill_between(epochs, mean - std, mean + std, color=line.get_color(), alpha=0.12)
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"{split} {label}")
        ax.grid(True, alpha=0.3)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
