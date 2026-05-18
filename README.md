# Hypergraph Neural Networks for Higher-Order Drug Interaction Modeling

Research codebase for modeling drug–drug interactions (DDI) with **hypergraph neural networks (HGNN)** and graph/MLP baselines. Drugs are nodes; multi-drug interactions are hyperedges. Pair-level prediction uses a learned scoring head on node embeddings.

> **Synthetic demo mode** (`source: synthetic_demo`) is for smoke tests only and is clearly labeled in code and configs. **Do not use it for publication results.** For research, provide your own [DrugBank](https://go.drugbank.com/) XML or TWOSIDES CSV under `data/raw/`.

## Project structure

```
hypergraph-ddi/
├── config/              # YAML experiment configs
├── data/
│   ├── raw/             # User-provided DrugBank / TWOSIDES files
│   ├── processed/       # Parquet + hypergraph NPZ
│   └── splits/          # Train/val/test JSON (edge-level, no leakage)
├── src/
│   ├── preprocessing/   # Loaders, normalization, hypergraph, splits
│   ├── datasets/        # PyTorch Dataset + PyG graph builder
│   ├── models/          # HGNN, GCN, GAT, GraphSAGE, MLP
│   ├── training/        # Trainer (BCE, Adam, early stopping)
│   ├── evaluation/      # Metrics + plots
│   ├── experiments/     # ExperimentRunner
│   └── ...
├── scripts/             # CLI entry points
└── experiments/         # Run outputs (logs, checkpoints, plots)
```

## Setup

```bash
cd hypergraph-ddi
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
export HYPERGRAPH_DDI_ROOT="$(pwd)"
```

**Note:** `torch-geometric` may require a PyTorch build matched to your CUDA/CPU setup. See [PyG installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Data preparation

### Synthetic demo (smoke test)

```bash
python scripts/preprocess.py --config config/demo_synthetic.yaml
```

### DrugBank

1. Download DrugBank full database XML (license required).
2. Place at `data/raw/drugbank.xml`.
3. Run:

```bash
python scripts/preprocess.py --config config/drugbank_hgnn.yaml
```

### TWOSIDES

1. Download TWOSIDES-style pair file from the original release (user-provided).
2. Place at `data/raw/twosides.csv` with columns configurable in YAML (`drug1_name`, `drug2_name` by default).
3. Run:

```bash
python scripts/preprocess.py --config config/twosides_gcn.yaml
```

**Splits:** interactions are split at the **hyperedge level** so the same interaction never appears in train and test.

**Hypergraph:** built from **training hyperedges only** to avoid test leakage into structure.

## Training

```bash
python scripts/train.py --config config/demo_synthetic.yaml
```

Outputs under `experiments/<name>_<run_id>/`:

- `checkpoints/best.pt`
- `train.log`
- `training_curves.png`
- `roc_curve.png`
- `test_metrics.json`

## Evaluation

```bash
python scripts/evaluate.py \
  --config config/demo_synthetic.yaml \
  --checkpoint experiments/demo_hgnn_smoke/checkpoints/best.pt \
  --output experiments/eval_smoke
```

## Multi-seed experiments

```bash
python scripts/run_experiment.py --config config/demo_synthetic.yaml --seeds 42 43 44
```

## Models

| Model | Config `model.name` | Structure used |
|-------|---------------------|----------------|
| HGNN | `hgnn` | Hypergraph incidence matrix H |
| MLP | `mlp` | Node features only |
| GCN | `gcn` | Pair-expanded train graph (PyG) |
| GAT | `gat` | Same |
| GraphSAGE | `graphsage` | Same |

## Metrics

ROC-AUC, PR-AUC, Precision, Recall, F1, Precision@K (K=10,50,100). Implemented in `src/evaluation/metrics.py`.

## Tests

```bash
pytest tests/test_smoke.py -v
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `HYPERGRAPH_DDI_ROOT` | Project root for resolving `./data/...` paths in configs |

## Limitations

- Node features default to random or identity vectors unless you plug in external drug descriptors (e.g. molecular fingerprints).
- DrugBank XML parsing depends on export format; verify parsed interaction counts.
- TWOSIDES column names vary by release; adjust `drug_a_col` / `drug_b_col` in config.
- Higher-order hyperedges (>2 drugs) are expanded to pairs for link prediction.
- No fabricated benchmark numbers; all reported metrics come from your data and runs.

## Citation

If you use this codebase, please cite your paper and acknowledge DrugBank / TWOSIDES data licenses.

## License

This project is licensed under the [MIT License](LICENSE).

**DrugBank and TWOSIDES data are subject to their own licenses** and are not covered by the MIT License above.
