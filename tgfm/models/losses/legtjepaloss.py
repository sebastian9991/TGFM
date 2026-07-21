"""LeGTJEPA objective.

    L = (1 - lambda_g - lambda_t) * L_cross
        + lambda_g * SIGReg(Z_g) + lambda_t * SIGReg(Z_t)

    L_cross = mean_i [ ||h_g(z_g_i) - sg(z_t_i)||^2 + ||h_t(z_t_i) - sg(z_g_i)||^2 ]

Both prediction targets are detached (stop-gradient): gradients reach each
encoder only through its own branch (LeVLJEPA Eq. 5-6). Note the target of
h_t is detached even though it comes from the graph encoder we are training,
and the target of h_g is detached even though the text *projection* is
trainable — the stop-gradient protects the trainable text projection from
being dragged toward the graph distribution by the h_g branch.

When the text side is fully frozen (``freeze_text_projection``), the h_t
branch and SIGReg(Z_t) have no trainable parameters upstream; both terms are
dropped and the objective degenerates to regression on fixed targets plus
SIGReg on the graph marginal.

Under DDP, embeddings are gathered across ranks before SIGReg so the
regularizer sees the global batch marginal (LeVLJEPA App. A computes SIGReg
with cross-device gathering); the per-sample cross loss needs no gathering.
"""

from typing import Dict

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.functional import mse_loss

from tgfm.models.sigreg import SIGReg
from tgfm.utils.args import LeGTJEPAArguments


def gather_embeddings(z: Tensor) -> Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return z
    from torch.distributed.nn.functional import all_gather

    return torch.cat(all_gather(z), dim=0)


class LeGTJEPALoss(torch.nn.Module):
    def __init__(self, args: LeGTJEPAArguments) -> None:
        super().__init__()
        self.lambda_g = args.lambda_graph
        self.lambda_t = 0.0 if args.freeze_text_projection else args.lambda_text
        self.text_branch_active = not args.freeze_text_projection
        self.sigreg = SIGReg(
            num_slices=args.sigreg_num_slices,
            num_quad_points=args.sigreg_num_quad_points,
            t_max=args.sigreg_t_max,
        )

    def forward(self, out: Dict[str, Tensor]) -> Dict[str, Tensor]:
        z_g, z_t = out['z_g'], out['z_t']

        # sum over dim, mean over batch: per-sample squared L2 as in Eq. 5.
        cross = mse_loss(out['pred_g'], z_t.detach(), reduction='none').sum(-1).mean()
        if self.text_branch_active:
            cross = (
                cross
                + mse_loss(out['pred_t'], z_g.detach(), reduction='none').sum(-1).mean()
            )

        sigreg_g = self.sigreg(gather_embeddings(z_g))
        sigreg_t = (
            self.sigreg(gather_embeddings(z_t))
            if self.lambda_t > 0
            else torch.zeros((), device=z_g.device)
        )

        total = (
            (1.0 - self.lambda_g - self.lambda_t) * cross
            + self.lambda_g * sigreg_g
            + self.lambda_t * sigreg_t
        )
        return {
            'loss': total,
            'cross': cross.detach(),
            'sigreg_graph': sigreg_g.detach(),
            'sigreg_text': sigreg_t.detach(),
        }
