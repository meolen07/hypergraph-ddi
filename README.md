# Hypergraph Neural Networks for Higher-Order Drug-Drug Interaction Modeling

Research codebase for modeling **drug–drug interactions (DDI)** with **hypergraph neural networks (HGNN)** and graph/MLP baselines. Drugs are nodes; multi-drug interactions are represented as **hyperedges**. Pair-level DDI prediction uses a learned scoring head on node embeddings.

**Repository:** [github.com/meolen07/hypergraph-ddi](https://github.com/meolen07/hypergraph-ddi)

This is an experimental research tool, not a clinical decision system. Performance depends heavily on data quality, feature choices, and evaluation protocol. Please read the [limitations](#limitations) and [disclaimer](#disclaimer) before using results.

---

## Overview

Many drug interactions involve more than two drugs at once (e.g., combination therapies, polypharmacy). Standard pairwise graphs treat each interaction as an edge between two nodes, which can under-represent **higher-order** structure.

This project builds a **hypergraph** from interaction records: each hyperedge connects all drugs involved in one interaction event. An HGNN propagates information along that higher-order structure; **GCN**, **GAT**, and **GraphSAGE** baselines operate on a pair-expanded graph derived from training data; an **MLP** baseline uses node features only.

The task is **binary link prediction** at the drug-pair level (positive pairs from interactions, negatives sampled per config). Hyperedges with more than two drugs are expanded to pairs for scoring, which is a deliberate simplification—see [limitations](#limitations).

---

## Motivation

- **Higher-order interactions:** Hypergraphs can represent multi-drug events without forcing a single pairwise edge to stand in for the full interaction.
- **Reproducible pipeline:** YAML configs, fixed seeds, edge-level splits, and a hypergraph built from **training hyperedges only** to reduce structural leakage.
- **Baselines for comparison:** The same preprocessing and metrics support HGNN and simpler models on the same splits.

The codebase is meant for **method exploration and ablation**, not as a turnkey benchmark leaderboard. You must run training on your own data to obtain meaningful numbers.

---

## Features

- **Data sources:** Synthetic demo (smoke tests), [DrugBank](https://go.drugbank.com/) XML, user-provided TWOSIDES-style CSV
- **Models:** HGNN, MLP, GCN, GAT, GraphSAGE (selected via `model.name` in config)
- **Preprocessing:** Loaders, normalization, hypergraph construction, edge-level train/val/test splits, negative sampling
- **Training:** BCE loss, Adam, early stopping on validation ROC-AUC (configurable)
- **Evaluation:** ROC-AUC, PR-AUC, precision, recall, F1, Precision@K (K = 10, 50, 100)
- **CLI scripts:** `preprocess`, `train`, `evaluate`, `run_experiment` (multi-seed)
- **Tests:** Pytest smoke test on synthetic demo only

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

---

## Limitations

- **Synthetic demo** is not representative of real pharmacology.
- **Default node features** are placeholders; without real drug descriptors, models mostly reflect graph structure and negatives sampling.
- **DrugBank parsing** depends on XML export format; always validate counts and spot-check interactions.
- **TWOSIDES** file layouts differ by release; column names must match your file.
- **Higher-order hyperedges** are expanded to pairs for link prediction; true multi-way effects are only partially captured.
- **No bundled SOTA numbers** in this README—all reported performance must come from your own runs.

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

## Author

**Huynh Mai Linh Nguyen** — research implementation and maintenance of this repository. Feedback and issues are welcome via GitHub.

---

## Disclaimer

**Do not fabricate or copy benchmark numbers into papers, slides, or portfolios without running this code on your data.** Metrics printed during a demo run (including synthetic smoke tests) are for debugging the pipeline only, not evidence of clinical or predictive utility.

This software is provided for research and education. It is **not** medical advice, not validated for patient care, and not a substitute for licensed databases, expert review, or regulatory processes. Always reproduce experiments locally and report limitations honestly.
