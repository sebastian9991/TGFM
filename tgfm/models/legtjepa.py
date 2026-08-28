"""LeGTJEPA model (three modalities: graph, text, image).

Graph side:  GraphGPS backbone (matched to GraphCLIP's Base configuration and
             data format: RWPE-32 in ``data.pe``, center node index in
             ``data.root_n_index``). ``graph_in_dim`` is the node-feature width,
             384 for GraphCLIP SBERT features, 768 for MM-Graph T5 features.
Text side:   two input modes, selected by ``text_input_mode``:
               'raw'     - frozen sentence-level LM backbone (MiniLM), mean
                           pooled over tokens. Used for the TAG datasets, where
                           node text is available as raw strings.
               'feature' - precomputed frozen text embeddings (MM-Graph ships
                           T5 features per node; there are no raw strings to
                           tokenize). No backbone is built; the projection maps
                           the stored ``text_in_dim`` vector into the shared
                           space, exactly parallel to the image tower.
Image side:  precomputed frozen image features (MM-Graph ships DINOv2 / CLIP /
             ViT / ImageBind vectors per node) projected into the shared space.
             Prefer a self-supervised feature (DINOv2) over a text-aligned one
             (CLIP / ImageBind): a text-aligned image feature pre-collapses the
             text-image volume before the objective acts on it.

Each tower feeds a one-hidden-layer MLP projection (Linear -> BatchNorm -> GELU
-> Dropout -> Linear) into the shared d-dimensional space where SIGReg is
applied. BatchNorm rather than LayerNorm is deliberate: a LayerNorm'd backbone
output blocks the SIGReg anti-collapse term (LeVLJEPA Sec. 3.2, LeWorldModel),
so the projection must re-expose the marginal to regularization. This applies
to precomputed features too -- they arrive from an encoder that ended in a
normalization, so the projection is doing the same job in 'feature' mode.

Cross-modal predictors h_g, h_t, h_image are depth-``predictor_depth`` MLPs
(LeVLJEPA App. A: depth 4, width 2048, BatchNorm, GELU, 10% dropout) trained
against stop-gradient targets. With three modalities a predictor's alignment
target is the set of the other two detached directions, handled in the loss
(LeGTJEPAVolumeLoss); the model exposes one predictor per tower.
"""

import logging
from typing import Any, Dict, Optional, Union

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

    def node_states(
        self, x: Tensor, pe: Tensor, edge_index: Tensor, batch: Tensor
    ) -> Tensor:
        """Per-node embeddings after the conv stack, before pooling. (N, hidden)."""
        x = torch.cat(
            (self.node_emb(x.squeeze(-1)), self.pe_lin(self.pe_norm(pe))), dim=1
        )
        for conv in self.convs:
            x = conv(x, edge_index, batch)
        return x

    def forward(
        self,
        x: Tensor,
        pe: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        center_idx: Tensor,
    ) -> Tensor:
        x = self.node_states(x, pe, edge_index, batch)
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
    # TODO: Why clamp? This was mentioned in the paper...
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class LeGTJEPA(torch.nn.Module):
    def __init__(self, args: LeGTJEPAArguments) -> None:
        super().__init__()
        self.args = args
        self.text_input_mode = getattr(args, 'text_input_mode', 'raw')
        if self.text_input_mode not in ('raw', 'feature'):
            raise ValueError(
                f'text_input_mode must be raw|feature, got {self.text_input_mode!r}'
            )
        self.use_image = getattr(args, 'use_image', False)

        self.graph_encoder = GraphGPSEncoder(args)

        if self.text_input_mode == 'raw':
            self.text_encoder = AutoModel.from_pretrained(args.text_model_id)
            text_in_dim = self.text_encoder.config.hidden_size
        else:
            # No backbone: the stored vector IS the encoder output.
            self.text_encoder = None
            text_in_dim = args.text_in_dim
            logging.info(
                'Text tower in feature mode: %d-d precomputed features.',
                text_in_dim,
            )

        d = args.embed_dim
        self.graph_projection = projection_head(
            self.graph_encoder.out_dim, args.proj_hidden_dim, d, args.proj_dropout
        )
        self.text_projection = projection_head(
            text_in_dim, args.proj_hidden_dim, d, args.proj_dropout
        )
        self.graph_predictor = predictor_mlp(
            d, args.predictor_hidden_dim, args.predictor_depth, args.predictor_dropout
        )
        self.text_predictor = predictor_mlp(
            d, args.predictor_hidden_dim, args.predictor_depth, args.predictor_dropout
        )

        if self.use_image:
            self.image_projection = projection_head(
                args.image_in_dim, args.proj_hidden_dim, d, args.proj_dropout
            )
            self.image_predictor = predictor_mlp(
                d,
                args.predictor_hidden_dim,
                args.predictor_depth,
                args.predictor_dropout,
            )

        if self.text_input_mode == 'raw' and args.freeze_text_backbone:
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
        if self.text_input_mode == 'raw' and self.args.freeze_text_backbone:
            # Keep frozen backbone in eval mode so its LayerNorm/dropout
            # statistics stay deterministic.
            self.text_encoder.eval()
        return self

    def encode_graph(self, batch: Any) -> Tensor:
        h = self.graph_encoder(
            batch.x, batch.pe, batch.edge_index, batch.batch, batch.root_n_index
        )
        return self.graph_projection(h)

    # def encode_node(self, batch: Any) -> Tensor:
    #     """Projected per-node embedding: conv stack, center-node row, no pool.
    #
    #     Runs the graph tower's node_states and returns the center node's row
    #     mapped into the shared space by graph_projection -- the representation
    #     the objective was trained on, at node rather than subgraph granularity.
    #     Each node must be encoded via its own ego-subgraph (item u centered on
    #     node u), so the center row is the only full-context row; see
    #     mm_graph_sampler. Returns (B, d).
    #     """
    #     h = self.graph_encoder.node_states(
    #         batch.x, batch.pe, batch.edge_index, batch.batch
    #     )
    #     return self.graph_projection(h[batch.root_n_index])
    #
    def encode_node(self, batch: Any) -> Tensor:
        """Per-node embedding: the pooled ego-subgraph embedding of node u.

        Each subgraph is centered on one node, so encode_graph's
        [mean-pool || center] output for that batch element is node u's
        contextual representation -- and it goes through graph_projection at
        the 2*hidden width the projection was trained on.
        """
        return self.encode_graph(batch)

    def encode_text(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Raw-text path: tokens -> frozen LM -> mean pool -> projection."""
        if self.text_input_mode != 'raw':
            raise RuntimeError(
                'encode_text requires text_input_mode="raw"; '
                'use encode_text_features in feature mode.'
            )
        if self.args.freeze_text_backbone:
            with torch.no_grad():
                out = self.text_encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                )
        else:
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pooling(out.last_hidden_state, attention_mask)
        return self.text_projection(pooled)

    def encode_text_features(self, text_x: Tensor) -> Tensor:
        """Feature path: precomputed (B, text_in_dim) vector -> projection."""
        return self.text_projection(text_x)

    def encode_image(self, image_x: Tensor) -> Tensor:
        """Precomputed (B, image_in_dim) image feature -> projection."""
        return self.image_projection(image_x)

    def _encode_text_any(self, batch_t: Union[Dict[str, Tensor], Tensor]) -> Tensor:
        """Dispatch on input type so forward is mode-agnostic."""
        if self.text_input_mode == 'feature':
            text_x = batch_t if isinstance(batch_t, Tensor) else batch_t['text_x']
            return self.encode_text_features(text_x)
        return self.encode_text(batch_t['input_ids'], batch_t['attention_mask'])

    def forward(
        self,
        batch_g: Any,
        batch_t: Union[Dict[str, Tensor], Tensor],
        image_x: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        z_g = self.encode_graph(batch_g)
        z_t = self._encode_text_any(batch_t)
        out = {
            'z_g': z_g,
            'z_t': z_t,
            'pred_g': self.graph_predictor(z_g),
            'pred_t': self.text_predictor(z_t),
        }
        if self.use_image:
            if image_x is None:
                raise ValueError('use_image=True but no image_x passed to forward')
            z_image = self.encode_image(image_x)
            out['z_image'] = z_image
            out['pred_image'] = self.image_predictor(z_image)
        return out

    def trainable_parameters(self) -> Any:
        return (p for p in self.parameters() if p.requires_grad)
