"""GraphGPS encoder for the graph LeJEPA experiment.

Input encoding follows the GraphGPS recipe:

    final_node_feat = concat(
        NodeEncoder(x),    # (N, node_out_dim)
        PEEncoder(pe),     # (N, pe_out_dim)    [optional]
        SEEncoder(se),     # (N, se_out_dim)    [optional]
    )                       # (N, dim)   where dim = node_out_dim + pe_out_dim + se_out_dim

`NodeEncoder`, `PEEncoder`, `SEEncoder` are passed in as `nn.Module`s.
They are *dataset-specific* — you define them where the dataset lives and hand
them to `GraphGPSEncoder` here. The encoder this file owns is just the
GraphGPS stack on top.

The contract per branch encoder:
    forward(raw_tensor) -> (N, out_dim)
    .out_dim attribute (int) so we can validate widths sum to `dim`.

For convenience this file also ships:
    - LinearEncoder:  one Linear layer; good default for x or PE.
    - MLPEncoder:     small MLP; good default for SE.
    - IdentityEncoder: pass-through; useful when you've already pre-projected.

These are *not* tied to specific datasets — they're generic primitives the
dataset-specific encoders can compose if you don't want to write a custom one.

Reference:
    Rampášek et al., "Recipe for a General, Powerful, Scalable Graph
    Transformer" (NeurIPS 2022).
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATConv, GCNConv, GINEConv, GPSConv


class IdentityEncoder(nn.Module):
    """Pass-through (with shape check). Use when the input already has out_dim width."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        if x.size(-1) != self.out_dim:
            raise ValueError(
                f'IdentityEncoder expected width {self.out_dim}, got {x.size(-1)}'
            )
        return x


class LinearEncoder(nn.Module):
    """Single Linear -> LayerNorm. Good baseline for node features or PEs."""

    def __init__(self, in_dim: int, out_dim: int, use_norm: bool = True):
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if use_norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.proj(x))


class MLPEncoder(nn.Module):
    """Small MLP with configurable depth. Good default for structural encodings."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        n_layers: int = 2,
        dropout: float = 0.0,
        use_norm: bool = True,
    ):
        super().__init__()
        self.out_dim = out_dim
        h = hidden_dim or out_dim

        layers: list[nn.Module] = []
        for i in range(n_layers):
            d_in = in_dim if i == 0 else h
            d_out = out_dim if i == n_layers - 1 else h
            layers.append(nn.Linear(d_in, d_out))
            if i < n_layers - 1:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim) if use_norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.mlp(x))


def _build_local_conv(
    local_gnn_type: str, dim: int, num_heads: int
) -> Optional[nn.Module]:
    """Construct the local MPNN that GPSConv will wrap.

    GPSConv accepts `conv=None` to disable the local branch (attention-only).
    """
    if local_gnn_type == 'None':
        return None
    if local_gnn_type == 'GINE':
        mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        return GINEConv(mlp, edge_dim=dim)
    if local_gnn_type == 'GAT':
        return GATConv(dim, dim // num_heads, heads=num_heads)
    if local_gnn_type == 'GCN':
        return GCNConv(dim, dim)
    raise ValueError(f'Unknown local_gnn_type: {local_gnn_type}')


class GraphGPSEncoder(nn.Module):
    """GraphGPS stack on top of composable input encoders.

    Args:
        node_encoder:    nn.Module mapping raw x -> (N, node_encoder.out_dim).
                         Typically dataset-specific (e.g. AtomEncoder for OGB,
                         Linear for continuous features, Embedding for atom types).
        pe_encoder:      Optional. Maps (N, raw_pe_dim) -> (N, pe_encoder.out_dim).
                         Pass None if the dataset has no PE.
        se_encoder:      Optional. Same shape contract, for structural encodings.
        dim:             Hidden width of the GPSConv stack. Must equal the sum
                         of out_dim across the provided branch encoders.
        num_layers:      Number of GPSConv blocks.
        num_heads:       Attention heads (and used to shape GAT local conv).
        dropout:         Dropout in GPSConv (attention + FFN).
        local_gnn_type:  "GINE" | "GAT" | "GCN" | "None".
        attn_type:       "multihead" | "performer" (per GPSConv).
        norm:            Intra-GPSConv norm ('batch_norm', 'layer_norm', ...).

    Forward inputs:
        x:           (N, raw_node_dim) raw node features.
        pe:          (N, raw_pe_dim) or None.
        se:          (N, raw_se_dim) or None.
        edge_index:  (2, E).
        batch:       (N,) graph assignment per node.
        edge_attr:   (E, ?), optional. Only used by GINE local conv.

    Returns:
        (N, dim) node-level embeddings. Pooling to view-level happens outside.

    Notes:
        - If a branch encoder is provided, the corresponding raw tensor must
          be provided at forward time (and vice versa). Mismatches raise.
        - We don't apply a top-level norm after the concat; LayerNorm comes
          for free inside the default LinearEncoder / MLPEncoder, and adding
          another can wash out per-branch scales. A final LayerNorm after the
          GPSConv stack is kept so downstream pooling sees stable scales.
    """

    def __init__(
        self,
        node_encoder: nn.Module,
        pe_encoder: Optional[nn.Module],
        se_encoder: Optional[nn.Module],
        dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
        local_gnn_type: str = 'GINE',
        attn_type: str = 'multihead',
        norm: str = 'batch_norm',
    ):
        super().__init__()

        # Validate widths sum to dim.
        node_w = self._get_out_dim(node_encoder, 'node_encoder')
        pe_w = (
            self._get_out_dim(pe_encoder, 'pe_encoder') if pe_encoder is not None else 0
        )
        se_w = (
            self._get_out_dim(se_encoder, 'se_encoder') if se_encoder is not None else 0
        )
        total = node_w + pe_w + se_w
        if total != dim:
            raise ValueError(
                f'Branch widths must sum to `dim`. Got node={node_w}, '
                f'pe={pe_w}, se={se_w} -> {total}, expected {dim}.'
            )

        self.dim = dim
        self.local_gnn_type = local_gnn_type

        self.node_encoder = node_encoder
        self.pe_encoder = pe_encoder
        self.se_encoder = se_encoder

        self.layers = nn.ModuleList(
            [
                GPSConv(
                    channels=dim,
                    conv=_build_local_conv(local_gnn_type, dim, num_heads),
                    heads=num_heads,
                    dropout=dropout,
                    attn_type=attn_type,
                    norm=norm,
                )
                for _ in range(num_layers)
            ]
        )

        self.out_norm = nn.LayerNorm(dim)

    @staticmethod
    def _get_out_dim(mod: nn.Module, name: str) -> int:
        if not hasattr(mod, 'out_dim'):
            raise AttributeError(
                f'{name} ({type(mod).__name__}) must expose an `out_dim` int '
                'attribute so widths can be validated against `dim`.'
            )
        assert isinstance(mod.out_dim, int)
        return int(mod.out_dim)

    def _encode_inputs(
        self,
        x: Tensor,
        pe: Optional[Tensor],
        se: Optional[Tensor],
    ) -> Tensor:
        parts = [self.node_encoder(x)]

        if self.pe_encoder is not None:
            if pe is None:
                raise ValueError('pe_encoder is set but `pe` was not provided.')
            parts.append(self.pe_encoder(pe))
        elif pe is not None:
            raise ValueError('`pe` was provided but no pe_encoder is configured.')

        if self.se_encoder is not None:
            if se is None:
                raise ValueError('se_encoder is set but `se` was not provided.')
            parts.append(self.se_encoder(se))
        elif se is not None:
            raise ValueError('`se` was provided but no se_encoder is configured.')

        return torch.cat(parts, dim=-1)

    def forward(
        self,
        x: Tensor,
        pe: Optional[Tensor],
        se: Optional[Tensor],
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        h = self._encode_inputs(x, pe, se)

        # GINE requires `edge_attr` of width `dim`. If none provided, fill zeros.
        kwargs = {}
        if self.local_gnn_type == 'GINE':
            if edge_attr is None:
                edge_attr = h.new_zeros(edge_index.size(1), self.dim)
            kwargs['edge_attr'] = edge_attr

        for layer in self.layers:
            h = layer(h, edge_index, batch, **kwargs)

        return self.out_norm(h)
