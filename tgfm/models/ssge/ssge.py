"""SSGE: Negative-Free Self-Supervised Gaussian Embedding of Graphs.

Reference: Liu, He, Zheng, Zhao. "Negative-Free Self-Supervised Gaussian
Embedding of Graphs." Neural Networks, 2024.
Original (DGL) implementation: https://github.com/Cloudy1225/SSGE

This module is the PyG re-implementation, written to plug into the same
codebase as the LeJEPA pretrainer. The structural correspondence is:

    SSGE                          <->   LeJEPA / LeGraph
    ----------------------------  --    -------------------------------
    single shared GCN/MLP         <->   single shared GraphGPS encoder
    two augmentations (G1, G2)    <->   two views (global / local)
    invariance loss (alignment)   <->   predictive loss   (out.pred)
    uniformity loss (W2 to N(0,I))<->   SIGReg loss        (out.sigreg)
    total = inv + lam * uni       <->   total = pred + lambd * sigreg

Unlike LeJEPA, SSGE operates *at the node level on the full graph*: each
augmentation preserves node identity (edge dropping + feature masking never
remove nodes), so the invariance term aligns node ``i`` across the two views,
and the uniformity term regularizes the distribution of the N node embeddings.
There is therefore no subgraph/view pooling step here.

The loss returns an ``SSGEOutput`` whose fields (``total``, ``pred``,
``sigreg``) mirror the ``LeJEPALoss`` output, so it is a drop-in for the
existing training/logging loop.
"""

from typing import Literal, NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

NormMode = Literal['batch', 'l2', 'batch+l2', 'none']

SSGE_PRESETS: dict[str, dict] = {
    'Cora': dict(
        lam=0.1,
        epochs=80,
        pd=0.3,
        pm=0.1,
        lr=1e-3,
        wd=1e-5,
        hid_dims=[256, 256],
        lr2=1e-2,
        wd2=1e-4,
        encoder='gcn',
    ),
    'CiteSeer': dict(
        lam=0.05,
        epochs=20,
        pd=0.4,
        pm=0.0,
        lr=1e-3,
        wd=1e-5,
        hid_dims=[512],
        lr2=1e-2,
        wd2=1e-2,
        encoder='gcn',
    ),
    'PubMed': dict(
        lam=0.6,
        epochs=100,
        pd=0.3,
        pm=0.5,
        lr=1e-3,
        wd=1e-5,
        hid_dims=[512, 256],
        lr2=1e-2,
        wd2=0.0,
        encoder='gcn',
    ),
    'WikiCS': dict(
        lam=0.5,
        epochs=50,
        pd=0.8,
        pm=0.1,
        lr=1e-2,
        wd=1e-6,
        hid_dims=[256, 256],
        lr2=1e-2,
        wd2=1e-4,
        encoder='gcn',
    ),
    'Computer': dict(
        lam=1.0,
        epochs=120,
        pd=0.1,
        pm=0.3,
        lr=1e-3,
        wd=1e-5,
        hid_dims=[512, 512],
        lr2=1e-2,
        wd2=1e-4,
        encoder='gcn',
    ),
    'CoauthorCS': dict(
        lam=0.05,
        epochs=80,
        pd=1.0,
        pm=0.2,
        lr=1e-3,
        wd=1e-5,
        hid_dims=[512, 512],
        lr2=1e-2,
        wd2=1e-4,
        encoder='mlp',
    ),
}


class SSGEOutput(NamedTuple):
    """Mirror of the LeJEPALoss output so this loss is a drop-in replacement.

    - ``pred``   : invariance / alignment term  (analog of predictive loss)
    - ``sigreg`` : uniformity term, W2 distance to N(0, I)  (analog of SIGReg)
    - ``total``  : pred + lambd * sigreg
    """

    total: Tensor
    pred: Tensor
    sigreg: Tensor


def batch_normalize(z: Tensor) -> Tensor:
    """Per-dimension standardization across the node (batch) axis.

    ``(z - mean) / std`` with unbiased std, exactly as in the reference.
    """
    return (z - z.mean(0)) / z.std(0)


def l2_normalize(z: Tensor, rescale: bool = False) -> Tensor:
    """Row-wise projection onto the sphere: z_i <- z_i / ||z_i||_2."""
    z = torch.nn.functional.normalize(z, p=2, dim=1)
    if rescale:
        z = z * z.shape[1] ** 0.5
    return z


def uniformity(z: Tensor) -> Tensor:
    """W2 distance between the empirical embedding law and N(0, I), up to consts.

    For a batch-normalized ``z`` the covariance ``C = z^T z / (n - 1)`` has
    (approximately) unit diagonal, so ``trace(C)`` is constant and the only
    data-dependent part of the squared Wasserstein-2 distance
    ``trace(C) + d - 2 * trace(sqrt(C))`` is ``-2 * sum(sqrt(eig(C)))``.
    """
    n = z.shape[0]
    cov = z.t() @ z / (n - 1)
    eigvals = torch.linalg.eigvalsh(cov)
    return -2.0 * torch.clamp(eigvals, min=1e-8).sqrt().sum()


class SSGELoss(nn.Module):
    """SSGE objective: invariance (alignment) + lambda * uniformity.

    Args:
        lambd: weight on the uniformity term. (Per-dataset values are in
            ``SSGE_PRESETS`` / the original ``params.txt``.)
        normalize: if True (default), batch-normalize each view inside the
            loss, matching the reference where ``normalize(Z)`` is applied
            before the loss is computed. Set False if you normalize upstream.
    """

    def __init__(self, lambd: float, norm_mode: NormMode = 'batch') -> None:
        super().__init__()
        self.lambd = lambd
        self.norm_mode = norm_mode

    def _apply_norm(self, z: Tensor) -> Tensor:
        if self.norm_mode in ('batch', 'batch+l2'):
            z = batch_normalize(z)
        if self.norm_mode in ('l2', 'batch+l2'):
            z = l2_normalize(z)

        return z

    def forward(self, z1: Tensor, z2: Tensor) -> SSGEOutput:
        z1 = self._apply_norm(z1)
        z2 = self._apply_norm(z2)

        # Invariance: maximize per-node agreement between the two views.
        # (negative dot product, averaged over nodes)
        inv = -(z1 * z2).sum() / z1.shape[0]

        # Uniformity: push each view toward an isotropic Gaussian.
        uni = 0.5 * (uniformity(z1) + uniformity(z2))

        total = inv + self.lambd * uni
        return SSGEOutput(total=total, pred=inv, sigreg=uni)
