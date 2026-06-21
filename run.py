"""Run the ZuCo text-only baseline sweep.

Trains every (model, head, mode) combination that doesn't already have a saved
result, so the sweep is resumable: existing runs are skipped, only missing ones
are computed. Each result lands in its own setup folder,
`<output-dir>/<head>_<mode>/<model>.json`.

Examples
--------
    # the whole grid (all models x {mean,cls,lstm} x {frozen,finetune})
    python run.py --output-dir results

    # just the lstm head, both modes, two backbones
    python run.py --head lstm --model-name bert-base-uncased roberta-base

    # recompute even if a result already exists
    python run.py --overwrite
"""

import argparse
import json
import os

from src.config import HEADS, MODELS, MODES, Config
from src.experiment import cross_validate, result_path, save_summary


def parse_args():
    p = argparse.ArgumentParser(description="Text-only BERT baseline sweep for ZuCo sentiment.")
    p.add_argument("--csv-path", default=Config.csv_path)
    p.add_argument("--model-name", nargs="+", default=MODELS,
                   help="one or more HuggingFace model ids")
    p.add_argument("--head", nargs="+", choices=HEADS, default=HEADS,
                   help="prediction heads to run")
    p.add_argument("--mode", nargs="+", choices=MODES, default=MODES,
                   help="frozen, finetune, or both")
    p.add_argument("--n-folds", type=int, default=Config.n_folds)
    p.add_argument("--max-length", type=int, default=Config.max_length)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--output-dir", default=Config.output_dir)
    p.add_argument("--overwrite", action="store_true",
                   help="recompute setups whose result already exists")
    p.add_argument("--epochs", type=int, default=None,
                   help="override the default epochs for every run in this sweep")
    return p.parse_args()


def _already_done(path, requested_epochs):
    """True if a usable result is already saved.

    Without an explicit `--epochs`, any existing file counts. With one, the file
    only counts if it was produced at that epoch budget, so bumping the epochs
    recomputes stale runs while still skipping the ones already at the new value.
    This makes an interrupted sweep safe to simply re-run.
    """
    if not os.path.exists(path):
        return False
    if requested_epochs is None:
        return True
    try:
        with open(path) as f:
            return json.load(f).get("epochs") == requested_epochs
    except (OSError, ValueError):
        return False


def main():
    args = parse_args()
    base = Config(
        csv_path=args.csv_path,
        n_folds=args.n_folds,
        max_length=args.max_length,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    done = []
    for model_name in args.model_name:
        for head in args.head:
            for mode in args.mode:
                path = result_path(args.output_dir, head, mode, model_name)
                if not args.overwrite and _already_done(path, args.epochs):
                    print(f"skip {head}/{mode} {model_name} (already in {path})")
                    done.append(path)
                    continue

                print(f"\n[{head}/{mode}] {model_name}")
                cfg = base.with_setup(head, mode)
                cfg.model_name = model_name
                if args.epochs is not None:
                    cfg.epochs = args.epochs
                summary = cross_validate(cfg)
                done.append(save_summary(summary, args.output_dir))

    _print_table(done)


def _print_table(paths):
    print("\n" + "=" * 74)
    print(f"{'head':<6}{'mode':<10}{'model':<30}{'accuracy':<14}{'macro-f1'}")
    print("-" * 74)
    for path in sorted(paths):
        with open(path) as f:
            s = json.load(f)
        print(
            f"{s.get('head', '?'):<6}{s['mode']:<10}{s['model_name']:<30}"
            f"{s['accuracy_mean']:.3f} +/-{s['accuracy_std']:.3f}  "
            f"{s['macro_f1_mean']:.3f} +/-{s['macro_f1_std']:.3f}"
        )


if __name__ == "__main__":
    main()
