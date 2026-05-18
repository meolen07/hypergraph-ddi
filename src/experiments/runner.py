"""Multi-run experiment orchestration with file logging."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from src.baselines.factory import build_model, make_trainer_forward
from src.datasets.ddi_dataset import DDIDataset, collate_ddi_batch, build_node_features
from src.datasets.graph_dataset import build_pyg_data, hyperedges_to_pair_edges
from src.evaluation.metrics import compute_metrics, get_roc_curve_data
from src.evaluation.visualization import plot_training_curves, plot_roc_curve
from src.preprocessing.hypergraph import HypergraphData, build_hypergraph
from src.preprocessing.negative_sampling import build_labeled_dataset
from src.preprocessing.splits import get_train_hyperedges, load_splits
from src.preprocessing.loaders import load_processed
from src.training.trainer import Trainer
from src.utils.config import load_config, resolve_paths
from src.utils.logging import setup_logger
from src.utils.seed import set_seed
from src.utils.paths import processed_dir, splits_dir


class ExperimentRunner:
    """Run training/evaluation from YAML config with reproducible seeds."""

    def __init__(self, config_path: str | Path) -> None:
        self.cfg = resolve_paths(load_config(config_path))
        self.config_path = Path(config_path)
        set_seed(int(self.cfg.get("seed", 42)))

    def _experiment_dir(self, run_id: str | None = None) -> Path:
        base = Path(self.cfg.get("experiment", {}).get("output_dir", "experiments"))
        if not base.is_absolute():
            base = Path(self.cfg["project_root"]) / base
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self.cfg.get("experiment", {}).get("name", "run")
        out = base / f"{name}_{run_id}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _load_data(self) -> Dict[str, Any]:
        proc = processed_dir(self.cfg)
        drugs_df, interactions = load_processed(proc)
        splits = load_splits(splits_dir(self.cfg))
        n_drugs = len(drugs_df)
        return {
            "drugs_df": drugs_df,
            "interactions": interactions,
            "splits": splits,
            "n_drugs": n_drugs,
        }

    def _prepare_tensors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        n_drugs = data["n_drugs"]
        mcfg = self.cfg.get("model", {})
        feat_dim = int(mcfg.get("in_dim", 64))
        x = build_node_features(n_drugs, feat_dim, seed=int(self.cfg.get("seed", 42)))
        device = torch.device(self.cfg.get("device", "cpu"))

        train_edges = get_train_hyperedges(data["splits"]["train"])
        hg = build_hypergraph(train_edges, n_nodes=n_drugs)
        hg_path = processed_dir(self.cfg) / "hypergraph"
        hg.save(hg_path)

        H = hg.to_torch_sparse(device)
        node_deg = torch.from_numpy(hg.node_deg).float().to(device)
        edge_deg = torch.from_numpy(hg.edge_deg).float().to(device)
        x = x.to(device)

        pair_edges = hyperedges_to_pair_edges(train_edges)
        pyg_data = build_pyg_data(n_drugs, pair_edges, x).to(device)

        return {
            "x": x,
            "H": H,
            "node_deg": node_deg,
            "edge_deg": edge_deg,
            "pyg_data": pyg_data,
            "hypergraph": hg,
        }

    def _build_loaders(self, data: Dict[str, Any]) -> Dict[str, DataLoader]:
        n_drugs = data["n_drugs"]
        x = build_node_features(
            n_drugs,
            int(self.cfg.get("model", {}).get("in_dim", 64)),
            seed=int(self.cfg.get("seed", 42)),
        )
        neg_ratio = int(self.cfg.get("data", {}).get("neg_per_pos", 1))
        train_pos = data["splits"]["train"]
        all_pos = (
            data["splits"]["train"]
            + data["splits"]["val"]
            + data["splits"]["test"]
        )
        loaders = {}
        bs = int(self.cfg.get("training", {}).get("batch_size", 128))
        for split_name in ("train", "val", "test"):
            exclude = None
            if split_name == "val":
                exclude = train_pos
            elif split_name == "test":
                exclude = all_pos
            labeled = build_labeled_dataset(
                data["splits"][split_name],
                n_neg_per_pos=neg_ratio,
                n_drugs=n_drugs,
                seed=int(self.cfg.get("seed", 42)) + hash(split_name) % 1000,
                exclude_positives=exclude,
            )
            ds = DDIDataset(labeled, x)
            loaders[split_name] = DataLoader(
                ds,
                batch_size=bs,
                shuffle=(split_name == "train"),
                collate_fn=collate_ddi_batch,
                num_workers=int(self.cfg.get("training", {}).get("num_workers", 0)),
            )
        return loaders

    def run(self, run_id: str | None = None) -> Dict[str, Any]:
        """Execute full train + test pipeline."""
        out_dir = self._experiment_dir(run_id)
        log_path = out_dir / "train.log"
        logger = setup_logger("experiment", log_path)
        logger.info(f"Experiment dir: {out_dir}")
        logger.info(f"Config: {self.config_path}")

        data = self._load_data()
        tensors = self._prepare_tensors(data)
        loaders = self._build_loaders(data)

        device = torch.device(self.cfg.get("device", "cpu"))
        model = build_model(self.cfg)
        train_extra = {k: tensors[k] for k in ("x", "H", "node_deg", "edge_deg", "pyg_data") if k in tensors}
        forward_fn = make_trainer_forward(model, self.cfg.get("model", {}).get("name", "hgnn"), train_extra)

        trainer = Trainer(model, self.cfg, device, forward_fn, logger=logger)
        fit_result = trainer.fit(
            loaders["train"],
            loaders["val"],
            out_dir / "checkpoints",
        )

        test_metrics = trainer.evaluate(loaders["test"])
        logger.info(f"Test metrics: {test_metrics}")

        # Save artifacts
        with open(out_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        with open(out_dir / "history.json", "w") as f:
            json.dump(fit_result["history"], f, indent=2)

        plot_training_curves(fit_result["history"], out_dir / "training_curves.png")

        # ROC on test set
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in loaders["test"]:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = forward_fn(batch)
                all_logits.append(logits.cpu())
                all_labels.append(batch["label"].cpu())
        import numpy as np
        y_true = torch.cat(all_labels).numpy()
        y_logits = torch.cat(all_logits).numpy()
        fpr, tpr = get_roc_curve_data(y_true, y_logits)
        plot_roc_curve(fpr, tpr, test_metrics["roc_auc"], out_dir / "roc_curve.png")

        return {
            "output_dir": str(out_dir),
            "test_metrics": test_metrics,
            "fit_result": fit_result,
        }
