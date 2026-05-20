# Synthetic benchmark results

- **Date:** 2026-05-20
- **Dataset:** synthetic_demo (50 drugs, 80 hyperedges)
- **Seeds:** 42, 43, 44, 45, 46
- **Hardware:** CPU (see local run logs under `experiments/`)

> **Note:** Synthetic demo data is for pipeline validation only — not pharmacological evidence.

## Metrics (test split, mean ± std over seeds)

| Model | ROC-AUC | PR-AUC | F1 |
|-------|---------|--------|-----|
| HGNN | 0.4893 ± 0.1034 | 0.7168 ± 0.0750 | 0.8235 ± 0.0000 |
| GCN | 0.5693 ± 0.0991 | 0.7549 ± 0.0722 | 0.3283 ± 0.4496 |
| GAT | 0.3824 ± 0.0850 | 0.6477 ± 0.0478 | 0.8235 ± 0.0000 |
| GraphSAGE | 0.4923 ± 0.0704 | 0.6909 ± 0.0419 | 0.8235 ± 0.0000 |
| MLP | 0.4970 ± 0.0998 | 0.7069 ± 0.0503 | 0.8235 ± 0.0000 |

## Reproduce

```bash
cd hypergraph-ddi
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
export HYPERGRAPH_DDI_ROOT="$(pwd)"
python scripts/run_synthetic_benchmark.py --seeds 42 43 44 45 46
```

Or per model:

```bash
python scripts/preprocess.py --config config/demo_synthetic.yaml
python scripts/run_experiment.py --config config/demo_synthetic.yaml --seeds 42 43 44 45 46 --skip-preprocess
```
