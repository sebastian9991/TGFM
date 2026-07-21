"""LeGTJEPA model.

Graph side:  GraphGPS backbone (matched to GraphCLIP's Base configuration and
             data format: 384-d SBERT node features, RWPE-32 in ``data.pe``,
             center node index in ``data.root_n_index``).
Text side:   frozen sentence-level LM backbone (MiniLM by default), mean-pooled.

Both backbones feed a one-hidden-layer MLP projection (Linear -> BatchNorm ->
GELU -> Dropout -> Linear) into the shared d-dimensional space where SIGReg is
applied. BatchNorm rather than LayerNorm is deliberate: MiniLM's final
LayerNorm output blocks the SIGReg anti-collapse term (LeVLJEPA Sec. 3.2,
LeWorldModel), so the projection must re-expose the marginal to regularization.

Cross-modal predictors h_g, h_t are depth-``predictor_depth`` MLPs
(LeVLJEPA App. A: depth 4, width 2048, BatchNorm, GELU, 10% dropout) trained
against stop-gradient targets.
"""

import logging
from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.nn import GELU, BatchNorm1d, Dropout, Linear, ModuleList, Sequential
from torch_geometric.nn import GPSConv, SAGEConv, global_mean_pool
from transformers import AutoModel

from tgfm.utils.args import LeGTJEPAArguments


class GraphGPSEncoder(torch.nn.Module):
    """GraphGPS backbone; forward signature matches GraphCLIP's GPS module."""

    def __init__(self, args: LeGTJEPAArguments) -> None:
        super().__init__()
        channels = args.graph_hidden_dim
        self.node_emb = Linear(args.graph_in_dim, channels - args.graph_pe_dim)
        self.pe_lin = Linear(32, args.graph_pe_dim)
        self.pe_norm = BatchNorm1d(32)
        self.convs = ModuleList(
            GPSConv(
                channels,
                SAGEConv(channels, channels),
                heads=8,
                attn_type=args.attn_type,
                attn_kwargs={'dropout': args.attn_dropout},
            )
            for _ in range(args.graph_num_layers)
        )
        # Pooled representation: [mean-pool || center node], as in GraphCLIP.
        self.out_dim = channels * 2

    def forward(
        self,
        x: Tensor,
        pe: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        center_idx: Tensor,
    ) -> Tensor:
        x = torch.cat(
            (self.node_emb(x.squeeze(-1)), self.pe_lin(self.pe_norm(pe))), dim=1
        )
        for conv in self.convs:
            x = conv(x, edge_index, batch)
        return torch.cat((global_mean_pool(x, batch), x[center_idx]), dim=1)


def projection_head(
    in_dim: int, hidden_dim: int, out_dim: int, dropout: float
) -> Sequential:
    """One-hidden-layer projection into the shared space (LeVLJEPA Sec. 3.2)."""
    return Sequential(
        Linear(in_dim, hidden_dim),
        BatchNorm1d(hidden_dim),
        GELU(),
        Dropout(dropout),
        Linear(hidden_dim, out_dim),
    )


def predictor_mlp(dim: int, hidden_dim: int, depth: int, dropout: float) -> Sequential:
    """Cross-modal predictor h: R^d -> R^d (LeVLJEPA App. A)."""
    if depth < 2:
        raise ValueError('predictor requires depth >= 2 (LeVLJEPA App. F, Fig. 2)')
    layers: list = []
    in_dim = dim
    for _ in range(depth):
        layers += [
            Linear(in_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            GELU(),
            Dropout(dropout),
        ]
        in_dim = hidden_dim
    layers.append(Linear(hidden_dim, dim))
    return Sequential(*layers)


def mean_pooling(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class LeGTJEPA(torch.nn.Module):
    def __init__(self, args: LeGTJEPAArguments) -> None:
        super().__init__()
        self.args = args

        self.graph_encoder = GraphGPSEncoder(args)
        self.text_encoder = AutoModel.from_pretrained(args.text_model_id)
        text_out_dim = self.text_encoder.config.hidden_size

        d = args.embed_dim
        self.graph_projection = projection_head(
            self.graph_encoder.out_dim, args.proj_hidden_dim, d, args.proj_dropout
        )
        self.text_projection = projection_head(
            text_out_dim, args.proj_hidden_dim, d, args.proj_dropout
        )

        self.graph_predictor = predictor_mlp(
            d, args.predictor_hidden_dim, args.predictor_depth, args.predictor_dropout
        )
        self.text_predictor = predictor_mlp(
            d, args.predictor_hidden_dim, args.predictor_depth, args.predictor_dropout
        )

        if args.freeze_text_backbone:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.text_encoder.eval()
            logging.info('Text backbone frozen: %s', args.text_model_id)
        if args.freeze_text_projection:
            for param in self.text_projection.parameters():
                param.requires_grad = False
            logging.info('Text projection frozen (fully locked text tower).')

    def train(self, mode: bool = True) -> 'LeGTJEPA':
        super().train(mode)
        if self.args.freeze_text_backbone:
            # Keep frozen backbone in eval mode so its LayerNorm/dropout
            # statistics stay deterministic.
            self.text_encoder.eval()
        return self

    def encode_graph(self, batch: Any) -> Tensor:
        h = self.graph_encoder(
            batch.x, batch.pe, batch.edge_index, batch.batch, batch.root_n_index
        )
        return self.graph_projection(h)

    def encode_text(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if self.args.freeze_text_backbone:
            with torch.no_grad():
                out = self.text_encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                )
        else:
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pooling(out.last_hidden_state, attention_mask)
        return self.text_projection(pooled)

    def forward(self, batch_g: Any, batch_t: Dict[str, Tensor]) -> Dict[str, Tensor]:
        z_g = self.encode_graph(batch_g)
        z_t = self.encode_text(batch_t['input_ids'], batch_t['attention_mask'])
        return {
            'z_g': z_g,
            'z_t': z_t,
            'pred_g': self.graph_predictor(z_g),  # predicts z_t
            'pred_t': self.text_predictor(z_t),  # predicts z_g
        }

    def trainable_parameters(self) -> Any:
        return (p for p in self.parameters() if p.requires_grad)
