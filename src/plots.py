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
