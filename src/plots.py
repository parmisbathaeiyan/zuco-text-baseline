"""Plotting helpers. Kept separate so the training code has no matplotlib pull."""

import matplotlib
matplotlib.use("Agg")  # write files without needing a display
import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix(cm, class_names, path, title=None):
    """Render a confusion matrix (rows = true, cols = predicted) to `path`.

    Cells show the raw count with the row-normalised rate underneath, so a
    glance tells you both how many sentences landed where and the per-class
    recall.
    """
    row_totals = cm.sum(axis=1, keepdims=True).clip(min=1)
    rates = cm / row_totals

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    if title:
        ax.set_title(title, fontsize=10)

    for i, j in np.ndindex(cm.shape):
        colour = "white" if rates[i, j] > 0.5 else "black"
        ax.text(j, i, f"{cm[i, j]}\n{rates[i, j]:.0%}",
                ha="center", va="center", color=colour, fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- comparison plots driven by the saved summaries --------------------------

# train / val / test share a colour; the split is told apart by line style.
_SPLIT_STYLE = {"train": "-", "val": "--", "test": ":"}
_METRICS = [("loss", "loss"), ("accuracy", "accuracy"), ("macro_f1", "macro-F1")]


def short_name(model_name):
    """Drop the org prefix so legends stay readable (sentence-transformers/...)."""
    return model_name.split("/")[-1]


def _curve(history, key):
    """Mean and std of a per-epoch metric across folds.

    `history` is the list of fold histories; every fold ran the same number of
    epochs, so the per-epoch values stack into a (folds, epochs) array.
    """
    stacked = np.array([[h[key] for h in fold] for fold in history])
    return stacked.mean(axis=0), stacked.std(axis=0)


def plot_learning_curves(summary, path):
    """Loss / accuracy / macro-F1 vs epoch, averaged over folds with std bands."""
    histories = [fold["history"] for fold in summary["folds"]]
    epochs = [h["epoch"] for h in histories[0]]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (metric, label) in zip(axes, _METRICS):
        for split, style in _SPLIT_STYLE.items():
            mean, std = _curve(histories, f"{split}_{metric}")
            line, = ax.plot(epochs, mean, style, label=split)
            ax.fill_between(epochs, mean - std, mean + std,
                            color=line.get_color(), alpha=0.15)
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].legend(frameon=False)
    fig.suptitle(f"{summary['mode']} - {short_name(summary['model_name'])}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_score_comparison(summaries, path):
    """Grouped bars of test accuracy and macro-F1 for every model and mode."""
    models = sorted({s["model_name"] for s in summaries}, key=short_name)
    modes = sorted({s["mode"] for s in summaries})
    lookup = {(s["model_name"], s["mode"]): s for s in summaries}

    x = np.arange(len(models))
    width = 0.8 / max(len(modes), 1)

    fig, axes = plt.subplots(1, 2, figsize=(max(8, 1.6 * len(models)), 4.5))
    for ax, metric, label in [(axes[0], "accuracy", "accuracy"),
                              (axes[1], "macro_f1", "macro-F1")]:
        for k, mode in enumerate(modes):
            means = [lookup[(m, mode)][f"{metric}_mean"] if (m, mode) in lookup else 0
                     for m in models]
            errs = [lookup[(m, mode)][f"{metric}_std"] if (m, mode) in lookup else 0
                    for m in models]
            offset = (k - (len(modes) - 1) / 2) * width
            ax.bar(x + offset, means, width, yerr=errs, capsize=3, label=mode)
        ax.set_xticks(x, [short_name(m) for m in models], rotation=30, ha="right")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(frameon=False, title="setup")
    fig.suptitle("Text-only baseline: test scores by backbone", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
