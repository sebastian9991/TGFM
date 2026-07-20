"""SIGReg pretraining for the GSTBench transfer pipeline.

Two-augmentation (GRACE-style) setup whose loss is delegated to the existing
LeJEPALoss module (centroid prediction + sliced Epps-Pulley SIGReg), used as
in the SSGE trainer: the two view embeddings are stacked along the view axis
as "global" views with an empty local set, so

    pred   = mean over (N, 2, d) of ||z_v - mu||^2,  mu = (z1 + z2) / 2
    sigreg = mean over the two view positions of the sliced Epps-Pulley
             statistic across the node axis
    total  = pred + lambd * sigreg          (beta=True)

Representation: no projector and no output activation — the loss acts
directly on the encoder output — and the input text embeddings (SBERT,
d_in) are residually connected to the final representation:

    z = f_theta(x, A) + x

The same skip is applied at inference, so the probed representation and the
loss representation coincide. The skip anchors z to the LLM feature space
(the encoder learns a correction to the text embedding rather than a
replacement for it), addressing the feature-destruction failure mode of the
earlier run. Requires d_in == hidden_dim (384 == 384 here); a Linear skip
projection would be needed otherwise.

Interface is identical to the other pretrain models in GSTBench
(PretrainGRACE, PretrainBGRL, ...):
    - __init__(encoder, device, args)
    - forward(data) -> scalar loss, where data = (x1, e1, x2, e2) as produced
      by Universal_Collator's two-view branch
    - inference(x, edges) -> (N, d) frozen embeddings for linear probing
    - trainable_parameters(), reset_parameters()
"""

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor

from tgfm.models.base_models.base_models import GATNet, GCNNet
from tgfm.models.leJepa_loss import LeJEPALoss, LeJEPALossOutput
from tgfm.utils.args import ModelArguments, TransferArguments

GraphEncoder = Union[GCNNet, GATNet]


def batch_normalize(z: Tensor) -> Tensor:
    """Per-dimension standardization across the node (batch) axis.

    ``(z - mean) / std`` with unbiased std, exactly as in the SSGE trainer.
    Currently unused; kept for ablation.
    """
    return (z - z.mean(0)) / z.std(0)


class PretrainSIGRegResidual(torch.nn.Module):
    """SIGReg pretraining, two-augmentation setup, input-feature residual."""

    encoder: GraphEncoder

    def __init__(
        self, encoder: GraphEncoder, device: torch.device, args: ModelArguments
    ) -> None:
        super().__init__()
        assert isinstance(args, TransferArguments)
        self.encoder = encoder
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

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.parameters())

    def _embed(self, x: Tensor, adj: SparseTensor) -> Tensor:
        """Encoder forward with input-feature residual: z = f(x, A) + x.

        Used identically by the training views and by inference, so the
        loss representation and the probed representation coincide.
        """
        z = self.encoder(x, adj)
        if z.shape[1] != x.shape[1]:
            raise ValueError(
                f'Input residual requires d_in == hidden_dim, '
                f'got d_in={x.shape[1]}, hidden_dim={z.shape[1]}. '
                f'Add a Linear skip projection to use mismatched dims.'
            )
        return z + x

    @torch.no_grad()
    def inference(self, x: Tensor, edges: Tensor) -> Tensor:
        """Frozen-encoder inference used by linear probing.

        Args:
            x: (N, d_in) node features.
            edges: (E, 2) edge list.

        Returns:
            (N, d) node embeddings, z = f(x, A) + x.
        """
        device = self.device
        self.eval()

        x, edges = x.to(device), edges.to(device)
        adj = SparseTensor.from_edge_index(
            edges.t().to(device),
            torch.ones(edges.shape[0]).to(device),
            [x.shape[0], x.shape[0]],
        )

        output = self._embed(x, adj)  # (N, d)

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

        # Residual views: z_v = f(x_v, A_v) + x_v. The skip uses the
        # AUGMENTED features (masked dims contribute zero), keeping each
        # view self-consistent.
        z1 = self._embed(x1, A1)
        z2 = self._embed(x2, A2)

        # Node axis plays the batch role: (N, 2, d) global views, no locals.
        z_global = torch.stack([z1, z2], dim=1)  # (N, 2, d)
        z_local = z1.new_empty(z1.size(0), 0, z1.size(1))  # (N, 0, d)

        out = self.loss_fn(z_global, z_local)
        self.last_output = out

        return out.total
