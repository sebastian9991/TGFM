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
    """Pairwise L_cross + SIGReg over the active modalities (graph, text, image).

    Two modalities (the original objective, unchanged):
        L_cross = ||h_g(z_g) - sg(z_t)||^2 + ||h_t(z_t) - sg(z_g)||^2

    Three modalities, the pairwise extension used as the baseline against the
    Gramian volume: every active anchor a predicts every other modality b,

        L_cross = sum_a sum_{b != a} mean_i ||h_a(z_i^a) - sg(z_i^b)||^2

    so k anchors x (k-1) targets = k(k-1) MSE terms, each per sample (no B^2
    term). One predictor per modality serves all of that modality's targets,
    matching the volume objective, where h_a's output is the single live column
    of A^(a). Anchors whose projection is frozen are dropped, as in the volume
    loss; their embeddings still serve as detached targets.
    """

    def __init__(self, args: LeGTJEPAArguments) -> None:
        super().__init__()
        self.use_text = getattr(args, 'use_text', True)
        self.use_image = getattr(args, 'use_image', False)
        self.text_branch_active = self.use_text and not args.freeze_text_projection
        self.image_branch_active = self.use_image and not getattr(
            args, 'freeze_image_projection', False
        )
        self.lambda_g = args.lambda_graph
        self.lambda_t = args.lambda_text if self.text_branch_active else 0.0
        self.lambda_i = (
            getattr(args, 'lambda_image', 0.0) if self.image_branch_active else 0.0
        )
        self.sigreg = SIGReg(
            num_slices=args.sigreg_num_slices,
            num_quad_points=args.sigreg_num_quad_points,
            t_max=args.sigreg_t_max,
        )

    def forward(self, out: Dict[str, Tensor]) -> Dict[str, Tensor]:
        z_g = out['z_g']
        zero = torch.zeros((), device=z_g.device)

        # (name, embedding, prediction, anchor active, SIGReg weight)
        modalities = [('graph', z_g, out['pred_g'], True, self.lambda_g)]
        if self.use_text:
            modalities.append(
                (
                    'text',
                    out['z_t'],
                    out.get('pred_t'),
                    self.text_branch_active,
                    self.lambda_t,
                )
            )
        if self.use_image:
            modalities.append(
                (
                    'image',
                    out['z_image'],
                    out.get('pred_image'),
                    self.image_branch_active,
                    self.lambda_i,
                )
            )

        # sum over dim, mean over batch: per-sample squared L2 as in Eq. 5.
        cross = zero
        for a, (_, _, pred_a, active, _) in enumerate(modalities):
            if not active:
                continue
            for b, (_, z_b, _, _, _) in enumerate(modalities):
                if b != a:
                    cross = (
                        cross
                        + mse_loss(pred_a, z_b.detach(), reduction='none')
                        .sum(-1)
                        .mean()
                    )

        lambda_sum = sum(m[4] for m in modalities)
        assert lambda_sum <= 1
        total = (1.0 - lambda_sum) * cross
        logs: Dict[str, Tensor] = {'cross': cross.detach()}
        for name, z, _, _, lam in modalities:
            sig = self.sigreg(gather_embeddings(z)) if lam > 0 else zero
            total = total + lam * sig
            logs[f'sigreg_{name}'] = sig.detach()
        if 'sigreg_text' not in logs:
            logs['sigreg_text'] = zero
        return {'loss': total} | logs
