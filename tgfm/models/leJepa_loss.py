"""LeJEPA loss for graphs.

Combines:
    1) Prediction loss: every view embedding pulled toward the per-subgraph
       centroid `mu_n` (mean over global views only, by default).
    2) SIGReg: anti-collapse regularization applied per view position across
       the batch, then averaged.


SIGReg: Sliced Isotropic Gaussian Regularization.

Reference:
    Balestriero & LeCun, "LeJEPA: Provable & Scalable Self-Supervised Learning ..." (2025).

Core idea:
    Project embeddings z (B, d) onto M random unit directions, getting (B, M)
    1-D distributions. For each direction, measure how far the empirical
    distribution is from a standard normal (here via a Cramer-von Mises style
    statistic on the sorted projections vs. the Gaussian CDF). Sum / average
    over directions.


Final loss:
    L = (1 - lambda) * pred_loss + lambda * sigreg_loss
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import scatter

from tgfm.models.multivariate.slicing import SlicingUnivariateTest
from tgfm.models.multivariate.univariate import EppsPulley


@dataclass
class LeJEPALossOutput:
    total: Tensor
    pred: Tensor
    sigreg: Tensor


class LeJEPALoss(nn.Module):
    """LeJEPA-style loss with centroid prediction + SIGReg.

    Args:
        lambd:      weight on SIGReg vs. prediction. Paper default ~0.05.
        num_slices: M for SIGReg.
        integration_points: The number of integration points.
        t_max: Upper bound on integration. As per original code default 3.0.
    Forward inputs:
        z_global: (B, V_g, d)
        z_local:  (B, V_l, d)
    """

    def __init__(
        self,
        lambd: float = 0.05,
        num_slices: int = 256,
        integration_points: int = 17,
        t_max: float = 3.0,
        beta: bool = True,
    ):
        super().__init__()
        self.lambd = lambd
        univariate_test = EppsPulley(t_max=t_max, n_points=integration_points)
        self.sigreg = SlicingUnivariateTest(
            univariate_test=univariate_test, num_slices=num_slices, reduction='mean'
        )
        self.beta = beta

    def forward(
        self,
        z_global: Tensor,  # (B, V_g, d)
        z_local: Tensor,  # (B, V_l, d)
    ) -> LeJEPALossOutput:
        B, V_g, d = z_global.shape
        _, V_l, _ = z_local.shape
        V = V_g + V_l

        mu = z_global.mean(dim=1)  # (B, d)

        all_views = torch.cat([z_global, z_local], dim=1)  # (B, V, d)
        # (B, V, d) - (B, 1, d) -> squared L2 per view, averaged over batch & view.
        diff = all_views - mu.unsqueeze(1)
        # pred_loss = (
        #     (diff**2).sum(dim=-1).mean()
        # )  # scalar, this will default to average over both dim=0,1
        pred_loss = (
            diff.square().mean()
        )  # mean over (B, v, d). According to Algorithm 2 of LeJEPA.

        # ---- SIGReg: per-view-position, averaged ----------------------------
        sigreg_loss = z_global.new_zeros(())
        for v in range(V):
            embeddings_v = all_views[:, v, :]  # (B, d)
            # Different seed per view position so M directions are decorrelated, this is handled internally in UnivariateTest
            sigreg_loss = sigreg_loss + self.sigreg(embeddings_v)
        sigreg_loss = sigreg_loss / V

        if self.beta:
            total = pred_loss + self.lambd * sigreg_loss
        else:
            total = (1.0 - self.lambd) * pred_loss + self.lambd * sigreg_loss
        return LeJEPALossOutput(
            total=total, pred=pred_loss.detach(), sigreg=sigreg_loss.detach()
        )


class NodeLevelLeJEPALoss(nn.Module):
    def __init__(
        self,
        lambd: float = 0.05,
        num_slices: int = 256,
        integration_points: int = 17,
        t_max: float = 3.0,
        normalize: bool = True,
        min_nodes: int = 16,
    ):
        super().__init__()
        self.lambd = lambd
        self.normalize = normalize
        self.min_nodes = min_nodes
        ut = EppsPulley(t_max=t_max, n_points=integration_points)
        self.sigreg = SlicingUnivariateTest(
            univariate_test=ut, num_slices=num_slices, reduction='mean'
        )

    def forward(
        self, h: Tensor, node_id: Tensor, view_id: Tensor, is_global_view: Tensor
    ) -> LeJEPALossOutput:
        # h:(M,d)  node_id:(M,)  view_id:(M,)  is_global_view:(num_views,) bool
        num_views = int(view_id.max().item()) + 1

        # (1) per-view batch-normalization (SSGE-style: standardize each view's block)
        # This may not be required.
        if self.normalize:
            h = (h - h.mean(0)) / (h.std(0) + 1e-5)

        # (2) SIGReg per view, over that view's node embeddings (many samples now)
        sigreg, counted = h.new_zeros(()), 0
        for v in range(num_views):
            zv = h[view_id == v]  # (n_v, d)
            if zv.size(0) < self.min_nodes:
                continue
            sigreg = sigreg + self.sigreg(zv) / zv.size(0)
            counted += 1
        sigreg = sigreg / max(counted, 1)

        # (3) predictive: per-node centroid from GLOBAL views (detached target)
        grow = is_global_view[view_id]  # (M,) bool
        if grow.any():
            g_h, g_id = h[grow], node_id[grow]
            uniq, inv = torch.unique(g_id, return_inverse=True)
            centroid = scatter(g_h, inv, dim=0, reduce='mean').detach()  # (U,d)
            pos = torch.searchsorted(uniq, node_id).clamp(max=uniq.numel() - 1)
            valid = uniq[pos] == node_id  # node seen in a global view
            diff = (h - centroid[pos])[valid]
            pred = (diff**2).sum(dim=-1).mean()
        else:
            pred = h.new_zeros(())

        total = (1.0 - self.lambd) * pred + self.lambd * sigreg
        return LeJEPALossOutput(total=total, pred=pred.detach(), sigreg=sigreg.detach())
