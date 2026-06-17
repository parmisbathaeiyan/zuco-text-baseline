"""Build plots for a single results folder (one pooling).

Reads every `*.json` in the results directory and writes, into a `plots/`
subfolder:

  * comparison.png           - test accuracy and macro-F1 across all backbones,
                               frozen vs fine-tuned
  * curves_by_model_<mode>.png - per setup, every backbone's test curves
                               (loss / accuracy / macro-F1) on shared axes
  * <run>_curves.png         - per-run learning curves with train, val and test

The per-run confusion matrices are written next to the JSONs by run.py.

Usage:
    python plot_results.py --results-dir results
"""

import argparse
import glob
import json
import os

from src.plots import plot_learning_curves, plot_model_curves, plot_score_comparison


def load_summaries(results_dir):
    summaries = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            summaries.append(json.load(f))
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Plot ZuCo baseline results.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default=None,
                        help="defaults to <results-dir>/plots")
    args = parser.parse_args()

    summaries = load_summaries(args.results_dir)
    if not summaries:
        raise SystemExit(f"no result JSONs found in {args.results_dir}")

    out_dir = args.out_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    written = 0

    # overall backbone comparison (accuracy + macro-F1, frozen vs fine-tuned)
    plot_score_comparison(summaries, os.path.join(out_dir, "comparison.png"))
    written += 1

    # per setup: all backbones' test curves side by side
    for mode in sorted({s["mode"] for s in summaries}):
        subset = [s for s in summaries if s["mode"] == mode]
        plot_model_curves(subset, os.path.join(out_dir, f"curves_by_model_{mode}.png"),
                          title=f"{mode}: test curves by backbone")
        written += 1

    # per run: the detailed train / val / test learning curves
    for s in summaries:
        stem = f"{s['mode']}_{s['model_name'].replace('/', '-')}_curves.png"
        plot_learning_curves(s, os.path.join(out_dir, stem))
        written += 1

    print(f"wrote {written} plots to {out_dir}")


if __name__ == "__main__":
    main()
