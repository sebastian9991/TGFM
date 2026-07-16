"""SIGReg pretraining for the GSTBench transfer pipeline.

Two-augmentation (GRACE-style) setup whose loss is delegated to the existing
LeJEPALoss module (centroid prediction + sliced Epps-Pulley SIGReg), used
exactly as in the SSGE trainer: the two view embeddings are stacked along the
view axis as "global" views with an empty local set, so

    pred   = mean over (N, 2, d) of ||h_v - mu||^2,  mu = (h1 + h2) / 2
    sigreg = mean over the two view positions of the sliced Epps-Pulley
             statistic across the node axis
    total  = pred + lambd * sigreg          (beta=True)

Interface is identical to the other pretrain models in GSTBench
(PretrainGRACE, PretrainDGI, ...):
    - __init__(encoder, device, args)
    - forward(data) -> scalar loss, where data = (x1, e1, x2, e2) as produced
      by Universal_Collator's two-view branch
    - inference(x, edges) -> (N, d) frozen embeddings for linear probing
    - trainable_parameters(), reset_parameters()
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_sparse import SparseTensor

from tgfm.models.base_models.base_models import GATNet, GCNNet
from tgfm.models.leJepa_loss import LeJEPALoss, LeJEPALossOutput
from tgfm.utils.args import ModelArguments, TransferArguments

GraphEncoder = Union[GCNNet, GATNet]


def batch_normalize(z: Tensor) -> Tensor:
    """Per-dimension standardization across the node (batch) axis.

    ``(z - mean) / std`` with unbiased std, exactly as in the SSGE trainer.
    """
    return (z - z.mean(0)) / z.std(0)


class PretrainSIGReg(torch.nn.Module):
    """SIGReg pretraining with a two-augmentation (GRACE-style) setup."""

    def __init__(
        self, encoder: GraphEncoder, device: torch.device, args: ModelArguments
    ) -> None:
        super(PretrainSIGReg, self).__init__()
        assert isinstance(args, TransferArguments)
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ELU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
        )
        self.act = nn.ReLU()
        self.device = device

        # LeJEPALoss owns the loss composition:
        #   lambda_sigreg -> lambd, n_directions -> num_slices (M),
        #   n_points -> integration_points (T for Epps-Pulley).
        self.loss_fn = LeJEPALoss(
            lambd=args.lambda_sigreg,
            num_slices=args.n_directions,
            integration_points=args.n_points,
        )

        # Last LeJEPALossOutput (pred / sigreg components, detached) for
        # inspection; the training loop only consumes the returned scalar.
        self.last_output: Optional[LeJEPALossOutput] = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        for module in self.projector:
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def trainable_parameters(self) -> list:
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.parameters())

    @torch.no_grad()
    def inference(self, x: Tensor, edges: Tensor) -> Tensor:
        """Frozen-encoder inference used by linear probing.

        Args:
            x: (N, d_in) node features.
            edges: (E, 2) edge list.

        Returns:
            (N, d) node embeddings.
        """
        device = self.device
        self.eval()

        x, edges = x.to(device), edges.to(device)
        adj = SparseTensor.from_edge_index(
            edges.t().to(device),
            torch.ones(edges.shape[0]).to(device),
            [x.shape[0], x.shape[0]],
        )

        output = self.encoder(x, adj)  # (N, d)

        return output  # (N, d)

    def forward(self, data: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        """Two-augmentation forward pass.

        Args:
            data: (x1, e1, x2, e2) — features (N, d_in) and edge lists (E_i, 2)
                  for the two augmented views.

        Returns:
            Scalar training loss (LeJEPALossOutput.total).
        """
        device = self.device
        x1, e1, x2, e2 = data
        x1, e1, x2, e2 = x1.to(device), e1.to(device), x2.to(device), e2.to(device)
        A1 = SparseTensor.from_edge_index(
            e1.t().to(device),
            torch.ones(e1.shape[0]).to(device),
            [x1.shape[0], x1.shape[0]],
        )
        A2 = SparseTensor.from_edge_index(
            e2.t().to(device),
            torch.ones(e2.shape[0]).to(device),
            [x2.shape[0], x2.shape[0]],
        )

        z1 = self.encoder(x1, A1)
        # z1 = self.act(z1)
        z2 = self.encoder(x2, A2)
        # z2 = self.act(z2)

        # h1 = self.projector(z1)
        # h2 = self.projector(z2)

        # Per-dimension standardization across nodes, as in the SSGE trainer.
        h1 = batch_normalize(z1)
        h2 = batch_normalize(z2)

        # Node axis plays the batch role: (N, 2, d) global views, no locals.
        z_global = torch.stack([h1, h2], dim=1)  # (N, 2, d)
        z_local = h1.new_empty(h1.size(0), 0, h1.size(1))  # (N, 0, d)

        out = self.loss_fn(z_global, z_local)
        self.last_output = out

        return out.total
