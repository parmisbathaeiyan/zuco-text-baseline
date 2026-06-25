"""Compare results across datasets, e.g. ZuCo vs TeCo.

Each `--dirs` entry is a results tree produced by run.py; `--labels` names them
(defaults to the folder names). For every (head, mode) present in more than one
tree it writes a grouped-bar chart -- x axis the backbone, bars the datasets --
into the output folder.

Usage:
    python compare_datasets.py \
        --dirs zuco_results_v2 teco_results_v2 --labels zuco teco --out zuco_vs_teco
"""

import argparse
import glob
import json
import os

from src.plots import head_of, plot_grouped_scores


def load_tagged(results_dir, label):
    summaries = []
    for path in sorted(glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)):
        if os.sep + "plots" + os.sep in path:
            continue
        with open(path) as f:
            s = json.load(f)
        s["dataset"] = label          # tag so plot_grouped_scores can group by it
        summaries.append(s)
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Compare ZuCo/TeCo result trees.")
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None,
                        help="one label per dir; defaults to the folder names")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(os.path.normpath(d)) for d in args.dirs]
    if len(labels) != len(args.dirs):
        raise SystemExit("number of --labels must match number of --dirs")

    summaries = []
    for label, d in zip(labels, args.dirs):
        tagged = load_tagged(d, label)
        if not tagged:
            raise SystemExit(f"no result JSONs found under {d}")
        summaries += tagged

    os.makedirs(args.out, exist_ok=True)
    heads = sorted({head_of(s) for s in summaries})
    modes = sorted({s["mode"] for s in summaries})

    for head in heads:
        for mode in modes:
            subset = [s for s in summaries if head_of(s) == head and s["mode"] == mode]
            if len({s["dataset"] for s in subset}) < 2:
                continue  # nothing to compare for this setup
            plot_grouped_scores(subset, "dataset",
                                os.path.join(args.out, f"datasets_{head}_{mode}.png"),
                                title=f"{head}/{mode}: {' vs '.join(labels)}")

    print(f"wrote dataset comparison plots for {', '.join(labels)} to {args.out}")


if __name__ == "__main__":
    main()
