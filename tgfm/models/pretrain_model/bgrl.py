"""BGRL pretraining for the GSTBench transfer pipeline.

Bootstrapped Graph Latent Learning (Thakoor et al., ICLR 2022,
https://arxiv.org/abs/2102.06514; official code: nerdslab/bgrl).
BYOL-style, negative-free: an online encoder + MLP predictor is trained to
predict the representations of an EMA target encoder across two augmented
views, with a symmetrized cosine loss

    L = 2 - cos(p(E_theta(v1)), sg(E_phi(v2))) - cos(p(E_theta(v2)), sg(E_phi(v1)))

where sg is stop-gradient and E_phi is the exponential moving average of
E_theta (momentum mm, optionally cosine-annealed to 1 as in the paper).

Deviations from the official implementation, kept deliberately for a
controlled GSTBench comparison:
    - Encoder is the shared GSTBench GCNNet/GATNet (no BatchNorm/PReLU
      encoder as in the BGRL repo) so architecture stays fixed across
      SSL methods.
    - Both views use the same augmentation rates (p_feat_drop, p_edge_drop
      from the collator); the original uses per-view asymmetric rates.

Interface is identical to the other pretrain models
(PretrainGRACE, PretrainSIGReg, ...):
    - __init__(encoder, device, args)
    - forward(data) -> scalar loss, data = (x1, e1, x2, e2) from
      Universal_Collator's two-view branch
    - inference(x, edges) -> (N, d) frozen ONLINE-encoder embeddings
    - trainable_parameters() -> online encoder + predictor only (the target
      encoder is EMA-updated, not optimized)
    - reset_parameters()

EMA scheduling: the target update runs at the START of each training forward,
which is equivalent to updating after the previous optimizer step and keeps
the GSTBench training loop unchanged. Under DDP the online weights are
identical across ranks after each step, so identical EMA updates keep the
(unsynced, requires_grad=False) target encoders identical too. Optionally
call set_total_steps(len(dataloader) * epochs) once in main() to enable the
paper's cosine momentum schedule; otherwise momentum is constant at
args.bgrl_mm.
"""

import copy
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor

from tgfm.models.base_models.base_models import GATNet, GCNNet
from tgfm.utils.args import ModelArguments, TransferArguments

GraphEncoder = Union[GCNNet, GATNet]


class MLPPredictor(nn.Module):
    """BGRL's predictor head: Linear -> BatchNorm -> PReLU -> Linear."""

    net: nn.Sequential

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.BatchNorm1d(hidden_size, momentum=0.01),
            nn.PReLU(1),
            nn.Linear(hidden_size, output_size, bias=True),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.net:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PretrainBGRL(torch.nn.Module):
    """BGRL pretraining with a two-augmentation setup and EMA target encoder."""

    encoder: GraphEncoder
    target_encoder: GraphEncoder
    predictor: MLPPredictor

    def __init__(
        self, encoder: GraphEncoder, device: torch.device, args: ModelArguments
    ) -> None:
        super().__init__()
        assert isinstance(args, TransferArguments)
        self.encoder = encoder
        self.predictor = MLPPredictor(
            args.hidden_dim, args.hidden_dim, hidden_size=args.bgrl_pred_hid
        )
        self.device = device

        # EMA target: a frozen copy of the online encoder. requires_grad=False
        # keeps its parameters out of DDP's reducer buckets, so no
        # find_unused_parameters error despite them never receiving grads.
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Momentum schedule state. With total_steps unset, momentum is
        # constant at mm_base; set_total_steps enables cosine annealing to 1.
        self.mm_base: float = args.bgrl_mm
        self.total_steps: Optional[int] = None
        self.step: int = 0

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        self.predictor.reset_parameters()
        # Target restarts as an exact copy of the online encoder.
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.step = 0

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        r"""Parameters updated by the optimizer: online encoder + predictor.

        The target encoder is intentionally excluded — it is updated via EMA.
        """
        return list(self.encoder.parameters()) + list(self.predictor.parameters())

    def set_total_steps(self, total_steps: int) -> None:
        """Enable the paper's cosine momentum schedule (mm_base -> 1)."""
        self.total_steps = total_steps

    def _target_momentum(self) -> float:
        if self.total_steps is None:
            return self.mm_base
        progress = min(self.step / max(self.total_steps, 1), 1.0)
        return 1.0 - (1.0 - self.mm_base) * (math.cos(math.pi * progress) + 1.0) / 2.0

    @torch.no_grad()
    def update_target_network(self) -> None:
        mm = self._target_momentum()
        for p_online, p_target in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            p_target.data.mul_(mm).add_(p_online.data, alpha=1.0 - mm)

    @torch.no_grad()
    def inference(self, x: Tensor, edges: Tensor) -> Tensor:
        """Frozen ONLINE-encoder inference used by linear probing.

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
        """Two-augmentation forward pass with symmetrized bootstrap loss.

        Args:
            data: (x1, e1, x2, e2) — features (N, d_in) and edge lists (E_i, 2)
                  for the two augmented views.

        Returns:
            Scalar training loss.
        """
        device = self.device

        # EMA update deferred from the previous optimizer step (see module
        # docstring); no-op on the very first step, where target == online.
        if self.training and self.step > 0:
            self.update_target_network()

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

        # Online branch: encoder + predictor on both views.
        q1 = self.predictor(self.encoder(x1, A1))
        q2 = self.predictor(self.encoder(x2, A2))

        # Target branch: EMA encoder, stop-gradient.
        with torch.no_grad():
            y1 = self.target_encoder(x1, A1).detach()
            y2 = self.target_encoder(x2, A2).detach()

        loss = (
            2.0
            - F.cosine_similarity(q1, y2, dim=-1).mean()
            - F.cosine_similarity(q2, y1, dim=-1).mean()
        )

        if self.training:
            self.step += 1

        return loss
