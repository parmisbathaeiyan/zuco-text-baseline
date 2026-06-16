# ZuCo text-only sentiment baseline

How far can you get on ZuCo Task 1 sentiment classification using **only the
sentence text** — no eye-tracking, no EEG? This repo answers that with a small,
self-contained BERT baseline so the number is easy to cite and easy to beat.

The data is the 400-sentence ZuCo Task 1 set, labelled negative / neutral /
positive. Because the set is small the scores are reported with **stratified
k-fold cross-validation** rather than a single split.

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
run.py              command line entry point
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

Per-fold and averaged scores print to the console, and a JSON summary for each
run lands in `results/`.

## Running on Colab

Open `notebooks/zuco_text_baseline.ipynb` in Colab (GPU runtime), run the cells
top to bottom. The first cell clones this repo; re-running it pulls the latest
code, so updates here show up in Colab with one cell.
