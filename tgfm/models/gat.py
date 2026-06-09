"""GAT architecture taken from Unigraph paper.
@article{he2025unigraph,
  title={UniGraph: Learning a Unified Cross-Domain Foundation Model for Text-Attributed Graphs},
  author={He, Yufei and Sui, Yuan and He, Xiaoxin and Hooi, Bryan},
  journal={ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year={2025}
}.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """Graph Attention Network using PyTorch Geometric."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        nhead: int = 8,
        nhead_out: int = 8,
        feat_drop: float = 0.2,
        attn_drop: float = 0.2,
        negative_slope: float = 0.2,
        concat_out: bool = False,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.concat_out = concat_out

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Build layers
        if num_layers == 1:
            # Single layer
            self.convs.append(
                GATConv(
                    in_dim,
                    out_dim,
                    heads=nhead_out,
                    dropout=feat_drop,
                    add_self_loops=True,
                    concat=concat_out,
                    negative_slope=negative_slope,
                )
            )
            self.norms.append(self._create_norm(out_dim, nhead_out, concat_out))

        else:
            # Input layer
            self.convs.append(
                GATConv(
                    in_dim,
                    hidden_dim,
                    heads=nhead,
                    dropout=feat_drop,
                    add_self_loops=True,
                    concat=True,
                    negative_slope=negative_slope,
                )
            )
            self.norms.append(self._create_norm(hidden_dim, nhead, True))

            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(
                        hidden_dim * nhead,
                        hidden_dim,
                        heads=nhead,
                        dropout=feat_drop,
                        add_self_loops=True,
                        concat=True,
                        negative_slope=negative_slope,
                    )
                )
                self.norms.append(self._create_norm(hidden_dim, nhead, True))

            # Output layer
            self.convs.append(
                GATConv(
                    hidden_dim * nhead,
                    out_dim,
                    heads=nhead_out,
                    dropout=feat_drop,
                    add_self_loops=True,
                    concat=concat_out,
                    negative_slope=negative_slope,
                )
            )
            self.norms.append(self._create_norm(out_dim, nhead_out, concat_out))

        # Activation functions
        self.activations = nn.ModuleList([nn.GELU() for _ in range(num_layers)])

        # Optional output head
        final_dim = out_dim * nhead_out if concat_out else out_dim
        self.head = nn.Linear(final_dim, out_dim)

        # Dropout
        self.dropout = nn.Dropout(attn_drop)

    def _create_norm(self, hidden_dim: int, num_heads: int, concat: bool) -> nn.Module:
        """Create normalization layer."""
        dim = hidden_dim * num_heads if concat else hidden_dim
        return nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            return_hidden: Whether to return concatenated hidden states

        Returns:
            Output embeddings, optionally with hidden states
        """
        hidden_list = [x]

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if self.norms[i] is not None:
                x = self.norms[i](x)

            # Apply activation
            if self.activations[i] is not None:
                x = self.activations[i](x)

            # Apply dropout (except last layer)
            if i < self.num_layers - 1:
                x = self.dropout(x)

            hidden_list.append(x)

        # Apply output head
        out = self.head(x)

        if return_hidden:
            # Concatenate all hidden states
            hidden_cat = torch.cat(hidden_list, dim=-1)
            return out, hidden_cat
        else:
            return out
