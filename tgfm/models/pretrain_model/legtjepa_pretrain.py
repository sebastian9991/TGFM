"""LeGTJEPA applied to the GSTBench pretraining pipeline.

Same model as ``tgfm.models.legtjepa.LeGTJEPA``, with two substitutions forced
by this pipeline:

    graph side  GraphGPS -> the shared GCN/GAT encoder, per node (no pooling,
                no RWPE, no center-node readout)
    text side   the LM backbone -> identity: node features here are already
                MiniLM sentence embeddings

The text projection and text predictor are kept. This is the
``freeze_text_backbone=True`` case of LeGTJEPA taken to its limit -- the
backbone is the identity, and the trainable projection is what makes
SIGReg(Z_t) an optimizable quantity at all.

Objective is unchanged and delegated to the existing ``LeGTJEPALoss``:

    L = (1 - lg - lt) * L_cross + lg * SIGReg(Z_g) + lt * SIGReg(Z_t)
    L_cross = mean ||h_g(z_g) - sg(z_t)||^2 + ||h_t(z_t) - sg(z_g)||^2

``inference`` returns the raw encoder output, so SIGReg shapes the projection
output while the linear probe reads the backbone.

Interface matches PretrainSIGReg / PretrainBGRL:
    __init__(encoder, device, args)
    forward(data) -> scalar loss
    inference(x, edges) -> (N, d)
    trainable_parameters()
    reset_parameters()
"""

import logging
from typing import Any, Dict

import torch
from torch import Tensor
from torch.nn import Module

from tgfm.models.legtjepa import predictor_mlp, projection_head
from tgfm.models.losses.legtjepaloss import LeGTJEPALoss
from tgfm.utils.args import TransferArguments


class PretrainLeGTJEPA(Module):
    def __init__(
        self, encoder: Module, device: torch.device, args: TransferArguments
    ) -> None:
        super().__init__()
        assert isinstance(args, TransferArguments)
        self.args = args
        self.device = device
        self.encoder = encoder

        d = args.embed_dim
        self.graph_projection = projection_head(
            args.hidden_dim, args.proj_hidden_dim, d, args.proj_dropout
        )
        self.text_projection = projection_head(
            args.text_input_dim, args.proj_hidden_dim, d, args.proj_dropout
        )
        self.graph_predictor = predictor_mlp(
            d, args.predictor_hidden_dim, args.predictor_depth, args.predictor_dropout
        )
        self.text_predictor = predictor_mlp(
            d, args.predictor_hidden_dim, args.predictor_depth, args.predictor_dropout
        )

        if args.freeze_text_projection:
            for param in self.text_projection.parameters():
                param.requires_grad = False
            # The h_t branch is dropped from the loss in this ablation, so the
            # text predictor never receives gradient; freeze it too or DDP with
            # find_unused_parameters=False crashes on it.
            for param in self.text_predictor.parameters():
                param.requires_grad = False
            logging.info('Text projection frozen (fully locked text tower).')

        self.criterion = LeGTJEPALoss(args)
        self.last_output: Dict[str, float] = {}

    def reset_parameters(self) -> None:
        for module in (
            self.encoder,
            self.graph_projection,
            self.text_projection,
            self.graph_predictor,
            self.text_predictor,
        ):
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
                continue
            for child in module.modules():
                if child is not module and hasattr(child, 'reset_parameters'):
                    child.reset_parameters()

    def trainable_parameters(self) -> Any:
        return (p for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _as_edge_index(edges: Tensor) -> Tensor:
        """PyG wants [2, E]; transfer_eval passes edge_index.t() = [E, 2]."""
        if edges.dim() == 2 and edges.size(0) != 2 and edges.size(1) == 2:
            edges = edges.t()
        return edges.contiguous()

    def _encode(self, x: Tensor, edges: Any) -> Tensor:
        return self.encoder(x, self._as_edge_index(edges))

    def encode_graph(self, x: Tensor, edges: Any) -> Tensor:
        return self.graph_projection(self._encode(x, edges))

    def encode_text(self, x: Tensor) -> Tensor:
        # Identity backbone: x is already the sentence embedding.
        return self.text_projection(x)

    def forward(self, data: Any) -> Tensor:
        x, edges = data[0], data[1]
        x = x.to(self.device)
        edges = edges.to(self.device)

        z_g = self.encode_graph(x, edges)
        z_t = self.encode_text(x)
        out = {
            'z_g': z_g,
            'z_t': z_t,
            'pred_g': self.graph_predictor(z_g),  # predicts z_t
            'pred_t': self.text_predictor(z_t),  # predicts z_g
        }
        losses = self.criterion(out)
        self.last_output = {k: float(v) for k, v in losses.items()}
        return losses['loss']

    @torch.no_grad()
    def inference(self, x: Tensor, edges: Any) -> Tensor:
        """Frozen representation for linear probing: the encoder output."""
        self.eval()
        return self.encode_graph(x.to(self.device), edges.to(self.device))
