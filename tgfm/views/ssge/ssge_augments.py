"""SSGE graph augmentations, ported to PyG.

Two stochastic augmentations are composed to build a view:
    - edge dropping : keep each edge with probability (1 - p_d)
    - feature masking: zero out whole feature columns with probability p_m

Both preserve node identity and node count, which is what lets the invariance
term align node ``i`` across the two views. This mirrors ``augment`` /
``drop_edge`` / ``mask_feat`` in the reference (which operate on a DGL graph;
here we operate directly on ``edge_index`` / ``x``).
"""

from typing import Tuple

import torch
from torch import Tensor


def drop_edge(edge_index: Tensor, drop_prob: float) -> Tensor:
    """Randomly drop edges.

    Each column of ``edge_index`` is kept independently with probability
    ``1 - drop_prob``. PyG edge_index for these datasets is already symmetric
    (both directions stored as separate columns), so dropping per-column
    matches the per-directed-edge drop of the reference on a bidirected graph.
    Self-loops are *not* added here; the ``GCNConv`` adds them at encode time.
    """
    if drop_prob <= 0.0:
        return edge_index
    num_edges = edge_index.size(1)
    keep = torch.bernoulli(
        torch.full((num_edges,), 1.0 - drop_prob, device=edge_index.device)
    ).bool()
    return edge_index[:, keep]


def mask_feature(x: Tensor, mask_prob: float) -> Tensor:
    """Randomly zero out whole feature columns with probability ``mask_prob``."""
    if mask_prob <= 0.0:
        return x
    drop_mask = (
        torch.empty(x.size(1), dtype=torch.float32, device=x.device).uniform_()
        < mask_prob
    )
    x = x.clone()
    x[:, drop_mask] = 0.0
    return x


def augment(
    x: Tensor,
    edge_index: Tensor,
    edge_drop_rate: float,
    feat_mask_rate: float,
) -> Tuple[Tensor, Tensor]:
    """Compose edge dropping and feature masking to produce one view."""
    edge_index = drop_edge(edge_index, edge_drop_rate)
    x = mask_feature(x, feat_mask_rate)
    return x, edge_index
