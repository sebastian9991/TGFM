from typing import Callable, Sequence

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv


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
