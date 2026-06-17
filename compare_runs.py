"""Compare two or more result folders, e.g. mean-pooled vs cls-pooled runs.

Each `--dirs` entry is a results folder produced by run.py; `--labels` names
them (defaults to the folder names). Writes to the output folder:

  * scores_accuracy.png    - grouped bars per setup, x = backbone, bars = label
  * scores_macro_f1.png    - same for macro-F1
  * delta_macro_f1.png     - second label minus first (only with two labels)
  * curves_<mode>.png      - test macro-F1 vs epoch, by backbone and label

Usage:
    python compare_runs.py \
        --dirs results_mean results_cls --labels mean cls --out results_compare
"""

import argparse
import glob
import json
import os

from src.plots import (
    plot_label_curves,
    plot_label_delta,
    plot_label_scores,
)


def load_summaries(results_dir):
    summaries = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            summaries.append(json.load(f))
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Compare ZuCo result folders.")
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None,
                        help="one label per dir; defaults to the folder names")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(os.path.normpath(d)) for d in args.dirs]
    if len(labels) != len(args.dirs):
        raise SystemExit("number of --labels must match number of --dirs")

    runs = {}
    for label, d in zip(labels, args.dirs):
        summaries = load_summaries(d)
        if not summaries:
            raise SystemExit(f"no result JSONs found in {d}")
        runs[label] = summaries

    os.makedirs(args.out, exist_ok=True)

    plot_label_scores(runs, "accuracy", os.path.join(args.out, "scores_accuracy.png"))
    plot_label_scores(runs, "macro_f1", os.path.join(args.out, "scores_macro_f1.png"))

    if len(runs) == 2:
        plot_label_delta(runs, "macro_f1", os.path.join(args.out, "delta_macro_f1.png"))

    for mode in sorted({s["mode"] for ss in runs.values() for s in ss}):
        plot_label_curves(runs, mode, os.path.join(args.out, f"curves_{mode}.png"))

    print(f"wrote comparison plots for {', '.join(labels)} to {args.out}")


if __name__ == "__main__":
    main()
