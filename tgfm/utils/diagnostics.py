"""Collapse diagnostics for SSL embeddings.

Tracks:
    - effective rank of the embedding covariance (proxy for dimensional collapse)
    - trace of covariance (total variance)
    - per-dim variance histogram (mean / min / max)
    - mean / std of pairwise cosine similarity (proxy for representational collapse)
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import Data


@dataclass
class CollapseStats:
    rank: float
    eff_rank: float  # exp(entropy(eigvals normalized))
    trace_cov: float
    var_min: float
    var_mean: float
    var_max: float
    cos_mean: float
    cos_std: float


@torch.no_grad()
def collapse_diagnostics(z: Tensor, rank_tol: float = 1e-5) -> CollapseStats:
    """Compute collapse-related diagnostics for embeddings z of shape (N, d).

    Args:
        z: (N, d) embeddings (any pooled set; typically the global-view
           embeddings flattened across the batch).
        rank_tol: relative tolerance for the hard rank.
    """
    N, d = z.shape
    z = z - z.mean(dim=0, keepdim=True)
    # Covariance via Gram on the feature axis.
    cov = (z.t() @ z) / max(N - 1, 1)  # (d, d)

    # Eigendecomp of symmetric PSD covariance.
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    total = eigvals.sum().clamp_min(1e-12)
    p = eigvals / total
    # Hard rank with tolerance.
    rank = int((eigvals > rank_tol * eigvals.max()).sum().item())
    # Effective rank = exp(entropy of normalized eigenvalues).
    entropy = -(p[p > 0] * p[p > 0].log()).sum()
    eff_rank = float(entropy.exp().item())

    per_dim_var = z.var(dim=0, unbiased=False)

    # Pairwise cosine similarities (sample if N is huge).
    n_sample = min(N, 1024)
    idx = torch.randperm(N, device=z.device)[:n_sample]
    zs = torch.nn.functional.normalize(z[idx], dim=-1)
    sim = zs @ zs.t()
    # Strip diagonal.
    mask = ~torch.eye(n_sample, dtype=torch.bool, device=z.device)
    off = sim[mask]

    return CollapseStats(
        rank=float(rank),
        eff_rank=eff_rank,
        trace_cov=float(cov.trace().item()),
        var_min=float(per_dim_var.min().item()),
        var_mean=float(per_dim_var.mean().item()),
        var_max=float(per_dim_var.max().item()),
        cos_mean=float(off.mean().item()),
        cos_std=float(off.std().item()),
    )


def _lift_view_nodes_to_global(view: Data, full_x: torch.Tensor) -> torch.Tensor:
    full_x.size(0)
    view.x.size(0)
    # Pairwise equality across feature dim; (|V_view|, N_full) bool matrix.
    matches = (view.x.unsqueeze(1) == full_x.unsqueeze(0)).all(dim=-1)
    # Each view row should match exactly one full row.
    counts = matches.sum(dim=1)
    assert (counts == 1).all(), (
        f'Feature lookup ambiguous: {(counts != 1).sum().item()} view rows '
        f'matched ≠1 full-graph rows. Possible feature collision.'
    )
    return matches.float().argmax(dim=1)


def global_local_containment(
    global_views: list,
    local_views: list,
    full_x: torch.Tensor,
) -> dict:
    """How much of each local view's node set is contained in each global view.

    For each (global G, local L) pair we compute:
        containment(G, L) = |nodes(G) ∩ nodes(L)| / |nodes(L)|

    This is asymmetric on purpose: it directly measures the failure mode
    flagged in the design notes -- "BFS global and METIS local (+ 1 hop)
    through a GNN may get the same information." If locals are (near-)subsets
    of globals, the prediction task degenerates because every local view is
    nearly a sub-pattern of the centroid's source.

    Returns:
        {
            "mean_containment": E_{G,L} [|G ∩ L| / |L|]            in [0, 1]
            "max_containment":  E_L [max_G |G ∩ L| / |L|]           in [0, 1]
            "mean_jaccard":     E_{G,L} [|G ∩ L| / |G ∪ L|]         in [0, 1]
            "pair_table":       (|G_views|, |L_views|) containment matrix
        }

    Interpretation:
        - mean_containment near 1.0  -> locals nested in globals (bad: degenerate task)
        - mean_containment near 0.0  -> locals disjoint from globals (also bad: nothing to learn)
        - mean_containment around 0.3-0.7 -> meaningful overlap with novel local detail
    """
    g_sets = [set(_lift_view_nodes_to_global(g, full_x).tolist()) for g in global_views]
    l_sets = [set(_lift_view_nodes_to_global(l, full_x).tolist()) for l in local_views]

    G, L = len(g_sets), len(l_sets)
    containment = torch.zeros(G, L)
    jaccard = torch.zeros(G, L)
    for i, g in enumerate(g_sets):
        for j, l in enumerate(l_sets):
            inter = len(g & l)
            union = len(g | l)
            containment[i, j] = inter / max(len(l), 1)
            jaccard[i, j] = inter / max(union, 1)

    return {
        'mean_containment': float(containment.mean().item()),
        'max_containment': float(containment.max(dim=0).values.mean().item()),
        'mean_jaccard': float(jaccard.mean().item()),
        'pair_table': containment,
    }
