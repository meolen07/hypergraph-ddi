# What if Drug Interactions Are Not Pairwise?

Most machine learning systems for drug–drug interaction (DDI) prediction assume a simple structure: interactions occur between pairs of drugs. This assumption enables the use of standard graph-based models such as GCNs and GATs, but it may not fully reflect how drug combinations behave in real biomedical settings.

In practice, many clinically relevant interactions involve **multiple drugs simultaneously**, where the effect emerges from the combination rather than any single pairwise relationship. This observation motivates a shift from graph-based modeling to **hypergraph-based representations**, where each interaction event can connect more than two drugs.

## The core idea

Instead of representing drug interactions as edges between two nodes, I model them as **hyperedges** in a hypergraph:

- Nodes represent drugs
- Hyperedges represent multi-drug interaction events

This formulation preserves higher-order structure that is typically lost when interactions are decomposed into pairs.

## What this repository provides

I built an experimental framework to study higher-order drug interaction modeling using **Hypergraph Neural Networks (HGNNs)**. The repository includes:

- Construction of hypergraph representations from drug interaction datasets
- A Hypergraph Neural Network for higher-order message passing
- Standard graph-based baselines, including GCN, GAT, and GraphSAGE
- A non-graph baseline (MLP) for reference
- A unified evaluation pipeline for fair comparison

The downstream task is formulated as **binary drug–drug interaction prediction**, evaluated using standard classification metrics.

## Why hypergraphs

Traditional graph neural networks rely on pairwise edges, which implicitly assume that interactions are decomposable into binary relationships. This can lead to:

- Loss of multi-drug contextual information
- Redundant representation of higher-order interactions
- Structural bias toward pairwise dependencies

Hypergraphs address this limitation by allowing a single interaction event to directly connect multiple drugs, preserving its full structure.

## Modeling approach

The Hypergraph Neural Network operates through two stages of message passing:

1. Aggregation from drug nodes to hyperedges
2. Propagation from hyperedges back to nodes

This enables each drug representation to incorporate both local and group-level interaction context. The learned embeddings are then used for pairwise interaction scoring.

## Experimental setup

The codebase supports evaluation on standard biomedical interaction datasets, including DrugBank and TWOSIDES-style data. Models are compared under a consistent pipeline with identical preprocessing, negative sampling strategy, and train/test splits.

Performance can be measured using:

- ROC-AUC
- PR-AUC
- Precision@K
- F1-score

**No benchmark numbers are reported in this README.** All metrics must come from running the training and evaluation scripts locally on your own data and hardware.

## Key insight

The main objective of this work is not only to improve predictive performance, but to investigate how **structural assumptions in data representation influence model behavior**.

By moving from pairwise graphs to hypergraphs, interaction events are encoded as higher-order objects rather than decomposed prematurely.

## Limitations

This study remains exploratory in nature:

- Node features are limited unless externally engineered
- Final prediction is still reduced to pairwise scoring
- Hypergraph construction depends on dataset quality and formatting
- No clinical validation is performed

Results should be interpreted as evidence about representation choices, not as clinical applicability.

## Summary

This project explores a simple but important question in representation learning:

> What happens if we model drug interactions as higher-order structures instead of pairwise relationships?

Hypergraph neural networks provide one possible answer by preserving interaction-level structure that standard graph models inherently discard.

---

## Repository

**hypergraph-ddi** — research codebase for modeling drug–drug interactions (DDI) with hypergraph neural networks and graph/MLP baselines.

**Repository:** [github.com/meolen07/hypergraph-ddi](https://github.com/meolen07/hypergraph-ddi)

**Author:** Huynh Mai Linh Nguyen — research implementation; feedback and issues are welcome on GitHub.

---

## Installation

**Requirements:** Python ≥ 3.9 (see `pyproject.toml`).

```bash
git clone https://github.com/meolen07/hypergraph-ddi.git
cd hypergraph-ddi

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .

export HYPERGRAPH_DDI_ROOT="$(pwd)"
```

**PyTorch Geometric:** `torch-geometric` must match your PyTorch and CUDA/CPU build. See the [official PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if `pip install` fails.

Optional sanity check (requires dependencies):

```bash
python scripts/verify_imports.py
```

---

## Project structure

```
hypergraph-ddi/
├── config/                 # YAML experiment configs
│   ├── demo_synthetic.yaml
│   ├── demo_mlp.yaml
│   ├── drugbank_hgnn.yaml
│   └── twosides_gcn.yaml
├── data/
│   ├── raw/                # User-provided DrugBank / TWOSIDES files
│   ├── processed/          # Parquet + hypergraph artifacts
│   └── splits/             # Train/val/test JSON (hyperedge-level)
├── src/
│   ├── preprocessing/      # Loaders, hypergraph, splits, negative sampling
│   ├── datasets/           # PyTorch Dataset + PyG graph builder
│   ├── models/             # HGNN, GCN, GAT, GraphSAGE, MLP
│   ├── training/           # Trainer
│   ├── evaluation/         # Metrics and plots
│   ├── experiments/        # ExperimentRunner
│   ├── inference/          # Checkpoint loading / prediction helpers
│   └── baselines/          # Model factory and forward wrappers
├── scripts/                # CLI entry points
├── tests/                  # Smoke tests
└── experiments/            # Run outputs (logs, checkpoints, plots)
```

---

## Quick start (synthetic demo)

The synthetic config (`source: synthetic_demo`) generates random drugs and hyperedges for **pipeline smoke tests only**. It is labeled in config and code; **do not use it for publication or benchmark claims.**

```bash
# 1. Preprocess (writes data/processed and data/splits)
python scripts/preprocess.py --config config/demo_synthetic.yaml

# 2. Train (writes experiments/demo_hgnn_<run_id>/)
python scripts/train.py --config config/demo_synthetic.yaml

# Optional: fixed run id (e.g. for tests)
python scripts/train.py --config config/demo_synthetic.yaml --run-id smoke
```

Artifacts under `experiments/<experiment.name>_<run_id>/` include:

- `checkpoints/best.pt`
- `train.log`
- `test_metrics.json`
- `history.json`
- `training_curves.png`
- `roc_curve.png`

MLP baseline on the same synthetic data:

```bash
python scripts/preprocess.py --config config/demo_mlp.yaml
python scripts/train.py --config config/demo_mlp.yaml
```

---

## Real data (DrugBank / TWOSIDES)

You must obtain and license data yourself. This repository does **not** redistribute DrugBank or TWOSIDES files.

### DrugBank

1. Download the DrugBank full database XML under their [license terms](https://go.drugbank.com/).
2. Place the file at `data/raw/drugbank.xml`.
3. Preprocess and train:

```bash
python scripts/preprocess.py --config config/drugbank_hgnn.yaml
python scripts/train.py --config config/drugbank_hgnn.yaml
```

Adjust paths and hyperparameters in `config/drugbank_hgnn.yaml` as needed. DrugBank XML structure can vary by export version—verify parsed interaction counts after preprocessing.

### TWOSIDES

1. Obtain a TWOSIDES-style pair file from the original data release (format varies by source).
2. Place it at `data/raw/twosides.csv` with columns matching your file (defaults: `drug1_name`, `drug2_name`).
3. Preprocess and train:

```bash
python scripts/preprocess.py --config config/twosides_gcn.yaml
python scripts/train.py --config config/twosides_gcn.yaml
```

Update `drug_a_col` and `drug_b_col` in the config if your CSV uses different column names.

### Split and hypergraph policy

- **Splits** are at the **hyperedge level**: the same interaction does not appear in both train and test.
- The **hypergraph** (and pair-expanded graph for GNN baselines) is built from **training hyperedges only**, so test interactions do not leak into structure used at train time.

---

## Training and evaluation

### Training

```bash
python scripts/train.py --config <path/to/config.yaml> [--run-id <id>]
```

Training reads processed data from `data/` (run `preprocess.py` first unless using `run_experiment.py`, which can preprocess automatically).

### Standalone evaluation

```bash
python scripts/evaluate.py \
  --config config/demo_synthetic.yaml \
  --checkpoint experiments/demo_hgnn_smoke/checkpoints/best.pt \
  --output experiments/eval_smoke
```

Writes `metrics.json` and `roc_curve.png` under `--output` when provided.

### Multi-seed runs

Runs preprocessing once (unless skipped), then trains one run per seed:

```bash
python scripts/run_experiment.py \
  --config config/demo_synthetic.yaml \
  --seeds 42 43 44
```

Summary written to `experiments/multi_seed_summary.json`. Use `--skip-preprocess` if data are already processed.

### Tests

```bash
pytest tests/test_smoke.py -v
```

---

## Models

Set `model.name` in your YAML config:

| Model       | `model.name` | Structure used                                      |
|------------|--------------|-----------------------------------------------------|
| HGNN       | `hgnn`       | Hypergraph incidence matrix **H** (training only) |
| MLP        | `mlp`        | Node features only                                  |
| GCN        | `gcn`        | Pair-expanded graph (PyG), training edges           |
| GAT        | `gat`        | Same as GCN                                         |
| GraphSAGE  | `graphsage`  | Same as GCN                                         |

Implementation: `src/models/`. HGNN supports optional attention (`use_attention` in config).

**Node features:** By default, features are random or identity vectors keyed by seed (`build_node_features`). For stronger results, plug in external descriptors (e.g., molecular fingerprints) in the dataset layer—this is left to the user.

---

## Metrics

Computed in `src/evaluation/metrics.py` on held-out pairs:

| Metric            | Key in JSON              |
|-------------------|--------------------------|
| ROC-AUC           | `roc_auc`                |
| PR-AUC            | `pr_auc`                 |
| Precision         | `precision` (threshold 0.5 on sigmoid logits) |
| Recall            | `recall`                 |
| F1                | `f1`                     |
| Precision@K       | `precision_at_10`, `precision_at_50`, `precision_at_100` |

If a split has only one class, some metrics return `nan`.

---

## Reproducibility

- Set `seed` in the YAML config; `src/utils/seed.py` fixes Python/NumPy/PyTorch seeds where applicable.
- Use the same `HYPERGRAPH_DDI_ROOT` and config file across preprocess, train, and evaluate.
- For variance estimates, use `scripts/run_experiment.py` with multiple `--seeds`.
- Save `test_metrics.json`, config copies, and checkpoints under `experiments/` for each run.

Document your data version, preprocessing choices, and hardware when reporting results.

**Additional scope notes:** The synthetic demo is not representative of real pharmacology; default node features are placeholders; DrugBank parsing depends on XML export format; TWOSIDES layouts vary by release; higher-order hyperedges are expanded to pairs for link prediction.

---

## License

This project is licensed under the [MIT License](LICENSE) (see also [LICENSE on GitHub](https://github.com/meolen07/hypergraph-ddi/blob/master/LICENSE)).

**DrugBank, TWOSIDES, and any third-party datasets remain under their own licenses** and are not included in or covered by this repository’s MIT license.

---

## Citation

If you use this codebase in published work, please cite it appropriately. A BibTeX entry can be added here when a formal publication is available:

```bibtex
@misc{hypergraph-ddi,
  author       = {Nguyen, Huynh Mai Linh},
  title        = {hypergraph-ddi: Hypergraph Neural Networks for Drug--Drug Interaction Modeling},
  year         = {2026},
  howpublished = {\url{https://github.com/meolen07/hypergraph-ddi}},
  note         = {Software. Replace with article citation when applicable.}
}
```

Please also cite **DrugBank** and **TWOSIDES** (or your data sources) according to their terms.

---

## Disclaimer

**Do not fabricate or copy benchmark numbers into papers, slides, or portfolios without running this code on your data.** Metrics printed during a demo run (including synthetic smoke tests) are for debugging the pipeline only, not evidence of clinical or predictive utility.

This software is provided for research and education. It is **not** medical advice, not validated for patient care, and not a substitute for licensed databases, expert review, or regulatory processes. Always reproduce experiments locally and report limitations honestly.
