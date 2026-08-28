"""MRR / Hits@K link-prediction evaluator, ranking against shipped negatives.

MM-Graph's lp-edge-split.pt provides, for each valid/test positive (u, v), a
fixed set of 150 negative targets in target_node_neg (shape (P, 150)). Each
positive's true target is ranked among {v} U neg(u), and the metrics are the
standard OGB-style ranking scores (Mosaic Sec. 3.2):

    MRR      mean reciprocal rank of the positive
    Hits@K   fraction of positives ranked in the top K

Higher is better for all three. Ranking uses the '>=' convention (a negative
scoring >= the positive counts against it), matching OGB's Evaluator so the
numbers line up with Mosaic's table.
"""

from typing import Callable, Dict

import torch
from torch import Tensor


@torch.no_grad()
def rank_metrics(pos_scores: Tensor, neg_scores: Tensor) -> Dict[str, float]:
    """pos_scores (P,), neg_scores (P, 150). Returns MRR, Hits@1, Hits@10."""
    # rank = 1 + (# negatives scoring >= the positive), per row.
    ge = (neg_scores >= pos_scores.unsqueeze(1)).sum(dim=1)
    ranks = ge + 1  # (P,), integer ranks in [1, 151]
    mrr = (1.0 / ranks.float()).mean().item()
    hits1 = (ranks <= 1).float().mean().item()
    hits10 = (ranks <= 10).float().mean().item()
    return {'mrr': mrr, 'hits@1': hits1, 'hits@10': hits10}


@torch.no_grad()
def evaluate_split(
    z: Tensor,
    source: Tensor,
    target: Tensor,
    target_neg: Tensor,
    decode: Callable,
    batch_size: int = 4096,
) -> Dict[str, float]:
    """Score every positive and its 150 negatives, batched over positives."""
    device = z.device
    pos_all, neg_all = [], []
    for i in range(0, source.numel(), batch_size):
        s = source[i : i + batch_size].to(device)
        t = target[i : i + batch_size].to(device)
        tneg = target_neg[i : i + batch_size].to(device)  # (b, 150)

        pos = decode(z, torch.stack([s, t]))  # (b,)
        b, k = tneg.shape
        s_rep = s.unsqueeze(1).expand(b, k).reshape(-1)
        neg = decode(z, torch.stack([s_rep, tneg.reshape(-1)])).view(b, k)

        pos_all.append(pos.cpu())
        neg_all.append(neg.cpu())

    return rank_metrics(torch.cat(pos_all), torch.cat(neg_all))
