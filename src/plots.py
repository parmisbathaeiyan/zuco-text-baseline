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


def plot_model_curves(summaries, path, title, split="test"):
    """One line per model across loss / accuracy / macro-F1, for a fixed setup.

    `summaries` should already be filtered to a single mode. Shows the chosen
    split (test by default) so the backbones can be compared on the same axes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (metric, label) in zip(axes, _METRICS):
        for s in summaries:
            histories = [fold["history"] for fold in s["folds"]]
            epochs = [h["epoch"] for h in histories[0]]
            mean, std = _curve(histories, f"{split}_{metric}")
            line, = ax.plot(epochs, mean, label=short_name(s["model_name"]))
            ax.fill_between(epochs, mean - std, mean + std,
                            color=line.get_color(), alpha=0.12)
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"{split} {label}")
        ax.grid(True, alpha=0.3)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- comparing two or more result sets (e.g. mean vs cls pooling) ------------

def _score(summary, metric):
    return summary[f"{metric}_mean"], summary[f"{metric}_std"]


def plot_label_scores(runs, metric, path):
    """Grouped bars comparing labelled result sets, one panel per setup.

    `runs` maps a label (e.g. "mean" / "cls") to its list of summaries. Within
    each setup panel the x axis is the backbone and the bars are the labels.
    """
    labels = list(runs)
    modes = sorted({s["mode"] for ss in runs.values() for s in ss})
    models = sorted({s["model_name"] for ss in runs.values() for s in ss}, key=short_name)
    lookup = {(lab, s["model_name"], s["mode"]): s
              for lab, ss in runs.items() for s in ss}

    x = np.arange(len(models))
    width = 0.8 / max(len(labels), 1)

    fig, axes = plt.subplots(1, len(modes), figsize=(max(7, 1.7 * len(models)) * len(modes) / 2, 4.5),
                             squeeze=False)
    for ax, mode in zip(axes[0], modes):
        for k, lab in enumerate(labels):
            means = [_score(lookup[(lab, m, mode)], metric)[0] if (lab, m, mode) in lookup else 0
                     for m in models]
            errs = [_score(lookup[(lab, m, mode)], metric)[1] if (lab, m, mode) in lookup else 0
                    for m in models]
            ax.bar(x + (k - (len(labels) - 1) / 2) * width, means, width,
                   yerr=errs, capsize=3, label=lab)
        ax.set_title(mode, fontsize=10)
        ax.set_xticks(x, [short_name(m) for m in models], rotation=30, ha="right")
        ax.set_ylabel(metric.replace("_", "-"))
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0][0].legend(frameon=False, title="pooling")
    fig.suptitle(f"{metric.replace('_', '-')} by pooling", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_label_delta(runs, metric, path):
    """Bar chart of the gap between exactly two labelled sets, per model and setup.

    Positive bars mean the second label beats the first on that metric.
    """
    labels = list(runs)
    if len(labels) != 2:
        raise ValueError("delta plot needs exactly two labelled result sets")
    a, b = labels
    modes = sorted({s["mode"] for ss in runs.values() for s in ss})
    models = sorted({s["model_name"] for ss in runs.values() for s in ss}, key=short_name)
    lookup = {(lab, s["model_name"], s["mode"]): s
              for lab, ss in runs.items() for s in ss}

    x = np.arange(len(models))
    width = 0.8 / max(len(modes), 1)

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(models)), 4.5))
    for k, mode in enumerate(modes):
        deltas = []
        for m in models:
            if (a, m, mode) in lookup and (b, m, mode) in lookup:
                deltas.append(_score(lookup[(b, m, mode)], metric)[0]
                              - _score(lookup[(a, m, mode)], metric)[0])
            else:
                deltas.append(0)
        ax.bar(x + (k - (len(modes) - 1) / 2) * width, deltas, width, label=mode)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [short_name(m) for m in models], rotation=30, ha="right")
    ax.set_ylabel(f"{b} - {a}  ({metric.replace('_', '-')})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, title="setup")
    fig.suptitle(f"Pooling gap: {b} minus {a}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_label_curves(runs, mode, path, split="test", metric="macro_f1"):
    """Overlay the test (or other split) curve for every model under each label.

    Colour identifies the backbone, line style identifies the label, so for a
    given setup you can read off how pooling changes each model's trajectory.
    """
    labels = list(runs)
    styles = ["-", "--", ":", "-."]
    label_style = {lab: styles[i % len(styles)] for i, lab in enumerate(labels)}

    models = sorted({s["model_name"] for ss in runs.values() for s in ss}, key=short_name)
    colour = {m: f"C{i}" for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for lab, summaries in runs.items():
        for s in summaries:
            if s["mode"] != mode:
                continue
            histories = [fold["history"] for fold in s["folds"]]
            epochs = [h["epoch"] for h in histories[0]]
            mean, _ = _curve(histories, f"{split}_{metric}")
            ax.plot(epochs, mean, label_style[lab], color=colour[s["model_name"]],
                    label=f"{short_name(s['model_name'])} [{lab}]")
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{split} {metric.replace('_', '-')}")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle(f"{mode}: {split} {metric.replace('_', '-')} by model and pooling", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
