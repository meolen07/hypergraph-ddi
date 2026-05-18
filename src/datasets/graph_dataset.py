"""PyTorch Geometric graph construction from hypergraph / pair edges."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch_geometric.data import Data


def build_pyg_data(
    n_nodes: int,
    edge_index_pairs: List[Tuple[int, int]],
    node_features: torch.Tensor,
) -> Data:
    """
    Build undirected PyG Data from pair list (for GCN/GAT/GraphSAGE).
    """
    if not edge_index_pairs:
        raise ValueError("No edges for PyG graph")
    src, dst = zip(*edge_index_pairs)
    edges = list(src) + list(dst)
    dsts = list(dst) + list(src)
    edge_index = torch.tensor([edges, dsts], dtype=torch.long)
    return Data(x=node_features.clone(), edge_index=edge_index, num_nodes=n_nodes)


def hyperedges_to_pair_edges(hyperedges: List[List[int]]) -> List[Tuple[int, int]]:
    """Expand hyperedges to pairwise edges for graph baselines."""
    pairs: set[Tuple[int, int]] = set()
    for nodes in hyperedges:
        nodes = sorted(set(nodes))
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                pairs.add((nodes[i], nodes[j]))
    return list(pairs)
