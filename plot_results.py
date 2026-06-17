"""Build all comparison plots from a results tree.

Reads every `*.json` under the results directory (it walks the per-setup
subfolders written by run.py) and writes, into a `plots/` subfolder:

  * overview_macro_f1.png / overview_accuracy.png
        heatmap of every backbone x setup (head/mode)
  * scores_by_head_<mode>.png
        per mode, each backbone's heads side by side (mean vs cls vs lstm)
  * scores_by_mode_<head>.png
        per head, frozen vs fine-tuned for each backbone
  * confusion_<head>_<mode>.png
        per setup, one confusion matrix per backbone
  * curves_<head>_<mode>.png
        per setup, every backbone's test curves (loss / accuracy / macro-F1)

Usage:
    python plot_results.py --results-dir results
"""

import argparse
import glob
import json
import os

from src.plots import (
    head_of,
    plot_confusion_grid,
    plot_grouped_scores,
    plot_model_curves,
    plot_overview_heatmap,
)


def load_summaries(results_dir):
    summaries = []
    for path in sorted(glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)):
        if os.sep + "plots" + os.sep in path:
            continue
        with open(path) as f:
            summaries.append(json.load(f))
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Plot ZuCo baseline results.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default=None, help="defaults to <results-dir>/plots")
    args = parser.parse_args()

    summaries = load_summaries(args.results_dir)
    if not summaries:
        raise SystemExit(f"no result JSONs found under {args.results_dir}")

    out = args.out_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(out, exist_ok=True)

    modes = sorted({s["mode"] for s in summaries})
    heads = sorted({head_of(s) for s in summaries})

    # overviews across the whole grid
    plot_overview_heatmap(summaries, "macro_f1", os.path.join(out, "overview_macro_f1.png"))
    plot_overview_heatmap(summaries, "accuracy", os.path.join(out, "overview_accuracy.png"))

    # heads compared within each mode
    for mode in modes:
        subset = [s for s in summaries if s["mode"] == mode]
        plot_grouped_scores(subset, "head", os.path.join(out, f"scores_by_head_{mode}.png"),
                            title=f"{mode}: heads compared")

    # frozen vs fine-tuned within each head
    for head in heads:
        subset = [s for s in summaries if head_of(s) == head]
        plot_grouped_scores(subset, "mode", os.path.join(out, f"scores_by_mode_{head}.png"),
                            title=f"{head} head: frozen vs fine-tuned")

    # per setup: confusion matrices and learning curves
    for head in heads:
        for mode in modes:
            subset = [s for s in summaries if head_of(s) == head and s["mode"] == mode]
            if not subset:
                continue
            plot_confusion_grid(subset, os.path.join(out, f"confusion_{head}_{mode}.png"),
                                title=f"{head}/{mode}: confusion matrices")
            plot_model_curves(subset, os.path.join(out, f"curves_{head}_{mode}.png"),
                              title=f"{head}/{mode}: test curves by backbone")

    print(f"wrote plots to {out}")


if __name__ == "__main__":
    main()
