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

from typing import Callable, NamedTuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv

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


class GCNEncoder(nn.Module):
    """Stack of GCN layers, activation on every layer except the last.

    Mirrors the original DGL ``GCN``: with ``hid_dims = [h1, h2, ...]`` the
    layer widths are ``[in_dim, h1, h2, ...]`` and the final layer has no
    activation. ``GCNConv`` adds self-loops and applies symmetric
    normalization internally, which matches DGL ``GraphConv(norm='both')``
    applied to a graph that has had self-loops added.
    """

    def __init__(
        self,
        in_dim: int,
        hid_dims: Sequence[int],
        act_fn: Callable[[Tensor], Tensor] = F.elu,
    ) -> None:
        super().__init__()
        self.act_fn = act_fn
        dims = [in_dim] + list(hid_dims)
        self.convs = nn.ModuleList(
            GCNConv(dims[i], dims[i + 1], add_self_loops=True, normalize=True)
            for i in range(len(dims) - 1)
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        last = len(self.convs) - 1
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != last:
                x = self.act_fn(x)
        return x


class MLPEncoder(nn.Module):
    """Two-layer MLP encoder (used for CoauthorCS in the paper).

    Ignores graph structure entirely. Matches the original DGL ``MLP``:
    ``Linear -> BatchNorm -> act -> Linear`` with widths
    ``in_dim -> hid_dims[0] -> hid_dims[-1]``.
    """

    def __init__(
        self,
        in_dim: int,
        hid_dims: Sequence[int],
        act_fn: Callable[[Tensor], Tensor] = F.elu,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hid_dims[0], bias=True)
        self.layer2 = nn.Linear(hid_dims[0], hid_dims[-1], bias=True)
        self.bn = nn.BatchNorm1d(hid_dims[0]) if use_bn else None
        self.act_fn = act_fn

    def forward(self, x: Tensor, edge_index: Tensor | None = None) -> Tensor:
        z = self.layer1(x)
        if self.bn is not None:
            z = self.bn(z)
        z = self.act_fn(z)
        z = self.layer2(z)
        return z


def build_encoder(
    in_dim: int,
    hid_dims: Sequence[int],
    kind: str = 'gcn',
    act_fn: Callable[[Tensor], Tensor] = F.elu,
) -> nn.Module:
    """Factory: ``kind`` is ``'gcn'`` (default) or ``'mlp'`` (CoauthorCS)."""
    kind = kind.lower()
    if kind == 'gcn':
        return GCNEncoder(in_dim, hid_dims, act_fn=act_fn)
    if kind == 'mlp':
        return MLPEncoder(in_dim, hid_dims, act_fn=act_fn)
    raise ValueError(f'Unknown encoder kind: {kind!r} (expected "gcn" or "mlp").')


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

    def __init__(self, lambd: float, normalize: bool = True) -> None:
        super().__init__()
        self.lambd = lambd
        self.normalize = normalize

    def forward(self, z1: Tensor, z2: Tensor) -> SSGEOutput:
        if self.normalize:
            z1 = batch_normalize(z1)
            z2 = batch_normalize(z2)

        # Invariance: maximize per-node agreement between the two views.
        # (negative dot product, averaged over nodes)
        inv = -(z1 * z2).sum() / z1.shape[0]

        # Uniformity: push each view toward an isotropic Gaussian.
        uni = 0.5 * (uniformity(z1) + uniformity(z2))

        total = inv + self.lambd * uni
        return SSGEOutput(total=total, pred=inv, sigreg=uni)
