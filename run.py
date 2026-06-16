"""Command line entry point for the ZuCo text-only sentiment baseline.

Examples
--------
    # both setups with the default backbone
    python run.py --mode both

    # just the fine-tuned RoBERTa
    python run.py --mode finetune --model-name roberta-base

    # sweep a few backbones, frozen and fine-tuned
    python run.py --mode both --model-name bert-base-uncased roberta-base distilbert-base-uncased
"""

import argparse

from src.config import Config
from src.experiment import cross_validate, save_summary


def parse_args():
    p = argparse.ArgumentParser(description="Text-only BERT baseline for ZuCo sentiment.")
    p.add_argument("--csv-path", default=Config.csv_path)
    p.add_argument("--model-name", nargs="+", default=["bert-base-uncased"],
                   help="one or more HuggingFace model ids to evaluate")
    p.add_argument("--mode", choices=["frozen", "finetune", "both"], default="both")
    p.add_argument("--pooling", choices=["mean", "cls"], default=Config.pooling)
    p.add_argument("--n-folds", type=int, default=Config.n_folds)
    p.add_argument("--max-length", type=int, default=Config.max_length)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--output-dir", default=Config.output_dir)
    # optional overrides for the per-mode optimisation defaults
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    return p.parse_args()


def build_config(args, model_name, mode):
    cfg = Config(
        csv_path=args.csv_path,
        model_name=model_name,
        pooling=args.pooling,
        n_folds=args.n_folds,
        max_length=args.max_length,
        seed=args.seed,
        output_dir=args.output_dir,
    ).with_mode(mode)

    # let explicit flags win over the per-mode defaults
    for field in ("epochs", "batch_size", "lr"):
        value = getattr(args, field)
        if value is not None:
            setattr(cfg, field, value)
    return cfg


def main():
    args = parse_args()
    modes = ["frozen", "finetune"] if args.mode == "both" else [args.mode]

    summaries = []
    for model_name in args.model_name:
        for mode in modes:
            print(f"\n[{mode}] {model_name}")
            cfg = build_config(args, model_name, mode)
            summary = cross_validate(cfg)
            save_summary(summary, cfg.output_dir)
            summaries.append(summary)

    print("\n" + "=" * 64)
    print(f"{'model':<28}{'mode':<10}{'accuracy':<14}{'macro-f1'}")
    print("-" * 64)
    for s in summaries:
        print(
            f"{s['model_name']:<28}{s['mode']:<10}"
            f"{s['accuracy_mean']:.3f} +/-{s['accuracy_std']:.3f}  "
            f"{s['macro_f1_mean']:.3f} +/-{s['macro_f1_std']:.3f}"
        )


if __name__ == "__main__":
    main()
