"""Build comparison plots from the JSON summaries written by run.py.

Reads every `*.json` in the results directory and writes, into a `plots/`
subfolder:

  * comparison.png        - test accuracy and macro-F1 across all backbones
  * <run>_curves.png      - per-run learning curves (loss / acc / F1, all splits)

The per-run confusion matrices are already saved next to the JSONs by run.py.

Usage:
    python plot_results.py --results-dir results
"""

import argparse
import glob
import json
import os

from src.plots import plot_learning_curves, plot_score_comparison


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

    plot_score_comparison(summaries, os.path.join(out_dir, "comparison.png"))
    for s in summaries:
        stem = f"{s['mode']}_{s['model_name'].replace('/', '-')}_curves.png"
        plot_learning_curves(s, os.path.join(out_dir, stem))

    print(f"wrote {len(summaries) + 1} plots to {out_dir}")


if __name__ == "__main__":
    main()
