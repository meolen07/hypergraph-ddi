"""Hypergraph construction: incidence matrix H and efficient storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class HypergraphData:
    """
    Hypergraph structure for HGNN.

    Attributes
    ----------
    n_nodes : number of drugs
    n_edges : number of hyperedges
    H : sparse incidence matrix (n_nodes x n_edges), H[v,e]=1 if node v in hyperedge e
  node_deg : diagonal node degree for normalization
    edge_deg : hyperedge degree (number of nodes per edge)
    """

    n_nodes: int
    n_edges: int
    H: sp.csr_matrix
    node_deg: np.ndarray
    edge_deg: np.ndarray

    def to_torch_sparse(self, device: torch.device | None = None) -> torch.Tensor:
        """Convert incidence matrix to torch sparse COO tensor."""
        coo = self.H.tocoo()
        indices = torch.from_numpy(
            np.vstack([coo.row, coo.col]).astype(np.int64)
        )
        values = torch.from_numpy(coo.data.astype(np.float32))
        shape = torch.Size(coo.shape)
        t = torch.sparse_coo_tensor(indices, values, shape)
        if device is not None:
            t = t.coalesce().to(device)
        return t.coalesce()

    def save(self, path: str | Path) -> None:
        """Save hypergraph to NPZ."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sp.save_npz(path / "H.npz", self.H)
        np.savez(
            path / "meta.npz",
            n_nodes=self.n_nodes,
            n_edges=self.n_edges,
            node_deg=self.node_deg,
            edge_deg=self.edge_deg,
        )

    @classmethod
    def load(cls, path: str | Path) -> "HypergraphData":
        path = Path(path)
        H = sp.load_npz(path / "H.npz")
        meta = np.load(path / "meta.npz")
        return cls(
            n_nodes=int(meta["n_nodes"]),
            n_edges=int(meta["n_edges"]),
            H=H,
            node_deg=meta["node_deg"],
            edge_deg=meta["edge_deg"],
        )


def build_hypergraph(
    hyperedges: List[List[int]],
    n_nodes: Optional[int] = None,
) -> HypergraphData:
    """
    Build hypergraph from list of hyperedges (each a list of node indices).

    Uses CSR sparse storage for memory efficiency.
    """
    if not hyperedges:
        raise ValueError("No hyperedges provided")

    if n_nodes is None:
        n_nodes = max(max(e) for e in hyperedges) + 1

    n_edges = len(hyperedges)
    rows, cols, data = [], [], []
    edge_sizes = np.zeros(n_edges, dtype=np.float64)

    for e_idx, nodes in enumerate(hyperedges):
        for v in set(nodes):
            if v < 0 or v >= n_nodes:
                raise ValueError(f"Node {v} out of range [0, {n_nodes})")
            rows.append(v)
            cols.append(e_idx)
            data.append(1.0)
        edge_sizes[e_idx] = len(set(nodes))

    H = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(n_nodes, n_edges),
        dtype=np.float32,
    )
    node_deg = np.array(H.sum(axis=1)).flatten()
    node_deg[node_deg == 0] = 1.0
    edge_deg = edge_sizes.copy()
    edge_deg[edge_deg == 0] = 1.0

    return HypergraphData(
        n_nodes=n_nodes,
        n_edges=n_edges,
        H=H,
        node_deg=node_deg,
        edge_deg=edge_deg,
    )


def hypergraph_laplacian_normalized(H: sp.csr_matrix) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Compute normalized hypergraph Laplacian components for HGNN."""
    dv = np.array(H.sum(axis=1)).flatten()
    de = np.array(H.sum(axis=0)).flatten()
    dv[dv == 0] = 1.0
    de[de == 0] = 1.0

    Dv_inv_sqrt = sp.diags(1.0 / np.sqrt(dv))
    De_inv = sp.diags(1.0 / de)
    H_de = H @ De_inv
    return H_de, dv, de
