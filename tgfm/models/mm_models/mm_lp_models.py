"""GNN encoders and dot-product link decoder for MM-Graph link prediction.

Reproduces the conventional-GNN rows of Mosaic Table 5: an encoder produces
node embeddings, a link (u, v) is scored by the dot product of the endpoint
embeddings, and training is BCE of positives against sampled negatives
(Mosaic Sec. 4.1). SAGE / GCN propagate over the graph; MLP is the same head
with no message passing, the structure-free baseline.

Input node features are the frozen LeGTJEPA representation emitted by
encode_node (projected, N x d), not the raw MM-Graph features -- that is the
axis this experiment varies against Table 5's CLIP / ImageBind rows.
"""

from typing import Any, List

import torch
from torch import Tensor
from torch.nn import Dropout, Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import GCNConv, SAGEConv


class SAGEEncoder(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.convs = ModuleList()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for a, b in zip(dims[:-1], dims[1:]):
            self.convs.append(SAGEConv(a, b))
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = torch.relu(x)
                x = torch.dropout(x, self.dropout, self.training)
        return x


class GCNEncoder(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.convs = ModuleList()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for a, b in zip(dims[:-1], dims[1:]):
            self.convs.append(GCNConv(a, b))
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = torch.relu(x)
                x = torch.dropout(x, self.dropout, self.training)
        return x


class MLPEncoder(torch.nn.Module):
    """No message passing: the structure-free baseline (Mosaic's MLP row)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: List = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for j, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(Linear(a, b))
            if j < len(dims) - 2:
                layers += [ReLU(), Dropout(dropout)]
        self.mlp = Sequential(*layers)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.mlp(x)  # edge_index ignored by design


ENCODERS = {'sage': SAGEEncoder, 'gcn': GCNEncoder, 'mlp': MLPEncoder}


def build_encoder(name: str, **kwargs: Any) -> torch.nn.Module:
    if name not in ENCODERS:
        raise ValueError(f'encoder must be one of {list(ENCODERS)}, got {name!r}')
    return ENCODERS[name](**kwargs)


def decode_link(z: Tensor, edge: Tensor) -> Tensor:
    """Dot-product score for edges (2, E): <z[u], z[v]> (Mosaic Sec. 4.1)."""
    return (z[edge[0]] * z[edge[1]]).sum(dim=-1)
