# ZuCo text-only sentiment baseline

How far can you get on ZuCo Task 1 sentiment classification using **only the
sentence text** — no eye-tracking, no EEG? This repo answers that with a small,
self-contained BERT baseline so the number is easy to cite and easy to beat.

The data is the 400-sentence ZuCo Task 1 set, labelled negative / neutral /
positive. Because the set is small the scores are reported with cross-validation
rather than a single split.

## Evaluation protocol

Splitting is nested so the reported number never sees the data it was tuned on:

- an **outer** stratified k-fold defines the **test** set for each fold,
- a stratified slice of the remaining data (`val_size`) is the **validation**
  set, used only to pick the best epoch,
- the model fits on **train**, and the score reported for the fold is on
  **test**, at the epoch chosen on validation.

Per-fold test predictions are pooled into one confusion matrix covering all 400
sentences exactly once. Each run also saves a full per-epoch history (train, val
and test loss / accuracy / macro-F1) so you can plot learning curves.

## Two setups

| setup | what trains | use it for |
|-------|-------------|------------|
| `frozen` | a linear probe on top of fixed encoder features | a lower bound: how linearly separable the off-the-shelf embeddings are |
| `finetune` | the whole encoder plus the head | the real text-only ceiling |

Both run through the exact same model and training code; they differ only by
whether the encoder weights are updated.

## Layout

```
src/config.py       hyper-parameters and label mapping
src/data.py         csv loading, tokenisation, stratified folds
src/model.py        encoder + pooling + linear head
src/engine.py       train / evaluate one fold
src/experiment.py   the cross-validation loop
src/plots.py        confusion matrices, learning curves, comparison bars
run.py              command line entry point
plot_results.py     build comparison plots from saved summaries
notebooks/          a Colab notebook that drives run.py
```

## Running locally

```bash
pip install -r requirements.txt
python run.py --mode both
```

Sweep a few backbones:

```bash
python run.py --mode both \
    --model-name bert-base-uncased roberta-base distilbert-base-uncased
```

Pooling defaults to a masked mean over the token embeddings; pass
`--pooling cls` to use the `[CLS]` token instead. Point each pooling at its own
`--output-dir` so the runs don't overwrite each other.

Per-fold and averaged scores print to the console, and a JSON summary for each
run lands in `results/`.

## Plots

Once a few runs have been saved, turn them into comparison figures:

```bash
python plot_results.py --results-dir results
```

This writes to `results/plots/`:

- `comparison.png` — test accuracy and macro-F1 across every backbone, frozen vs
  fine-tuned, with std error bars over folds
- `<run>_curves.png` — per-run learning curves (loss / accuracy / macro-F1 for
  train, val and test, averaged over folds with std bands)
- `<run>_confusion.png` — per-run confusion matrix, written by `run.py`

The results directory itself holds only the JSON summaries; every PNG lives in
`results/plots/`.

## Running on Colab

Open `notebooks/zuco_text_baseline.ipynb` in Colab (GPU runtime), run the cells
top to bottom. The first cell clones this repo; re-running it pulls the latest
code, so updates here show up in Colab with one cell.
