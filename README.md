# ZuCo text-only sentiment baseline

How far can you get on Task 1 sentiment classification using **only the sentence
text** — no eye-tracking, no EEG? This repo answers that with small,
self-contained baselines so the numbers are easy to cite and easy to beat.

Two datasets, same task and pipeline:
- **ZuCo** — 400 English sentences (negative / neutral / positive).
- **TeCo** — 165 Persian sentences (the verified translations used in the TeCo
  EEG recordings), same three labels.

Because the sets are small, scores come from cross-validation rather than a
single split.

## Setups

A run is one combination of a **backbone**, a **head**, and a **mode**.

| head | what sits on the encoder |
|------|--------------------------|
| `mean` | masked mean of the token embeddings + a linear classifier |
| `cls` | the `[CLS]` token + a linear classifier |
| `lstm` | the text component from Hollenstein et al. (2021): a BiLSTM over the token sequence, flattened, then dense + dropout + softmax |

| mode | encoder |
|------|---------|
| `frozen` | weights fixed, only the head trains |
| `finetune` | encoder and head train together |

The full grid is therefore `backbones x {mean, cls, lstm} x {frozen, finetune}`.
The `lstm` head reproduces the reference architecture (lstm_dim 256, dense 128,
dropout 0.3); the paper fine-tunes BERT, here it can be run frozen or fine-tuned
like the others.

## Evaluation protocol

Splitting is nested so the reported number never sees the data it was tuned on:

- an **outer** stratified k-fold defines the **test** set for each fold,
- a stratified slice of the remaining data (`val_size`) is the **validation**
  set, used only to pick the best epoch,
- the model fits on **train**, and the score reported is on **test**.

Per-fold test predictions are pooled into one confusion matrix over all 400
sentences. Each run also saves a per-epoch history (train / val / test loss,
accuracy and macro-F1).

## Layout

```
src/config.py       backbones, heads, modes and their hyper-parameters
src/data.py         csv loading, tokenisation, stratified folds
src/model.py        encoder + mean / cls / lstm head
src/engine.py       train / evaluate one fold
src/experiment.py   the cross-validation loop and result paths
src/plots.py        confusion matrices, score bars, overview, curves
run.py              the sweep (resumable, skips existing results)
plot_results.py     build every comparison plot from a results tree
compare_datasets.py put two datasets (ZuCo vs TeCo) side by side
notebooks/          a Colab notebook that drives the whole thing
data/               zuco and teco sentence csvs (same schema)
```

## Results layout

Each setup writes to its own folder, one JSON per backbone:

```
<output-dir>/
  mean_frozen/bert-base-uncased.json
  mean_finetune/bert-base-uncased.json
  cls_frozen/...
  lstm_frozen/...
  plots/                 # everything plot_results.py produces
```

## Running

```bash
pip install -r requirements.txt

# the whole grid for a dataset; already-finished runs are skipped
python run.py --dataset zuco --output-dir zuco_results
python run.py --dataset teco --output-dir teco_results

# a slice
python run.py --dataset teco --head lstm --mode frozen --model-name bert-base-multilingual-cased

# rebuild a dataset's plots, then compare the two datasets
python plot_results.py --results-dir zuco_results
python compare_datasets.py --dirs zuco_results teco_results --labels zuco teco --out zuco_vs_teco
```

`--dataset` selects the csv (`zuco` is the default); both csvs live in `data/`.
TeCo is Persian, so the multilingual backbones (`bert-base-multilingual-cased`,
`sentence-transformers/LaBSE`) are the meaningful ones there — the English-only
models are kept for the sweep but score near chance on Persian.

`run.py` is resumable: it skips any `<head>_<mode>/<model>.json` that already
exists, so you can fill the grid incrementally (pass `--overwrite` to force a
recompute).

## Colab

Open `notebooks/zuco_text_baseline.ipynb` on a GPU runtime and run top to bottom.
The first cell clones the repo, and re-running it pulls the latest code. There is
an optional cell to fold existing result folders into the new layout so they are
skipped.
