"""LeGTJEPA objective with a Gramian volume alignment term.

    L = (1 - sum_a lambda_a) / (k * B) * sum_a sum_i det G_i^(a)
        + sum_a lambda_a * SIGReg(Z^(a))

For sample ``i`` and anchor modality ``a``, the columns of A_i^(a) are the k
modality directions, with the anchor column replaced by that modality's
prediction and the other k-1 columns detached:

    A_i^(a) = [ unit(h_a(z_i^(a))) | sg(unit(z_i^(b))) for b != a ]   (d x k)
    G_i^(a) = A_i^(a).T @ A_i^(a)                                     (k x k)

det G_i^(a) is the squared volume of the k-parallelotope spanned by those
columns (Gantmacher, Matrix Theory, 1959; Thm. 1 of Cicchetti et al., GRAM,
ICLR 2025). Columns are unit-norm so diag(G) = 1 and det G in [0, 1]: 1 when
the directions are mutually orthogonal, 0 when they are collinear.

This is a drop-in replacement for ``LeGTJEPALoss``. The only change relative
to that objective is the alignment term; the stop-gradient structure, the
per-modality SIGReg terms, and the (1 - lambda_tot) / lambda_a convex
weighting are identical.

Relation to the MSE alignment term it replaces. For k = 2,

    det G = 1 - <unit(h_g(z_g)), sg(unit(z_t))>^2 = sin^2(theta)

(GRAM App. A.3), against ||h_g(z_g) - sg(z_t)||^2 = ||h||^2 + ||z_t||^2 -
2||h|| ||z_t|| cos(theta). The volume term is scale-invariant in both
arguments and keeps only the angle. Two consequences to keep in mind when
reading the runs:
  * det G is even in cos(theta), so theta = pi is a minimum alongside
    theta = 0, and d/dcos(1 - cos^2) = -2cos vanishes at orthogonality --
    which is where randomly initialised embeddings sit, |cos| = O(d^-1/2).
    Expect a slower start than the MSE term and watch ``cos_align``.
  * det G in [0, 1] while the MSE term is O(d). lambda_graph / lambda_text
    tuned for the MSE objective will not transfer; re-sweep them.

Numerics. det G is evaluated by the closed-form cofactor expansion for
k = 2, 3, which is the definition written out and keeps the code readable;
k >= 4 falls back to ``torch.linalg.det``. Checked on this batch shape: the
two agree to float32 precision, both give finite gradients at exactly
singular G, and the closed form is ~3x faster at k=2 and on par at k=3, so
the choice is transparency and speed rather than stability.
Round-off can push det G a few units in the last place below zero near
convergence; the loss is left unclamped there (the gradient is still
correct) and only the logged volume is clamped.

Under DDP, embeddings are gathered across ranks before SIGReg so the
regularizer sees the global batch marginal (LeVLJEPA App. A). The alignment
term is per-sample and needs no gathering.
"""

from typing import Dict, List

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.functional import normalize

from tgfm.models.sigreg import SIGReg
from tgfm.utils.args import LeGTJEPAArguments


def gather_embeddings(z: Tensor) -> Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return z
    from torch.distributed.nn.functional import all_gather

    return torch.cat(all_gather(z), dim=0)


def gramian(G: Tensor) -> Tensor:
    """Det G for a batch of k x k Gram matrices. ``G`` is (..., k, k) -> (...).

    Cofactor expansion along the first row, written out for the k = 2 and
    k = 3 cases this project uses; larger k defers to ``torch.linalg.det``.
    """
    k = G.shape[-1]
    if k == 2:
        a, b = G[..., 0, 0], G[..., 0, 1]
        c, d = G[..., 1, 0], G[..., 1, 1]
        return a * d - b * c
    if k == 3:
        a, b, c = G[..., 0, 0], G[..., 0, 1], G[..., 0, 2]
        d, e, f = G[..., 1, 0], G[..., 1, 1], G[..., 1, 2]
        g, h, i = G[..., 2, 0], G[..., 2, 1], G[..., 2, 2]
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return torch.linalg.det(G)


def squared_volumes(
    embeddings: List[Tensor], predictions: List[Tensor], anchor: int
) -> Tensor:
    """Det G_i^(a) for every sample i in the batch, for a fixed anchor a.

    ``embeddings[b]`` and ``predictions[b]`` are (B, d) tensors for modality b.
    Only ``predictions[anchor]`` carries gradient; every other column is
    detached, so gradients reach each encoder solely through its own branch
    (LeVLJEPA Eq. 5).
    """
    columns = []
    for b, z_b in enumerate(embeddings):
        v = predictions[anchor] if b == anchor else z_b.detach()
        columns.append(normalize(v.float(), dim=-1, eps=1e-8))

    A = torch.stack(columns, dim=-1)  # (B, d, k)
    G = A.transpose(-2, -1) @ A  # (B, k, k), diag(G) = 1
    return gramian(G)  # (B,)


class LeGTJEPAVolumeLoss(torch.nn.Module):
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
        embeddings = [z_g, z_t]
        predictions = [out['pred_g'], out['pred_t']]

        # One term per anchor modality. With the text side frozen the h_t
        # branch has no trainable parameters upstream and is dropped, exactly
        # as in the MSE objective; the 1/k average then runs over the anchors
        # that are actually present.
        anchors = [0, 1] if self.text_branch_active else [0]
        det_per_anchor = [squared_volumes(embeddings, predictions, a) for a in anchors]
        align = torch.stack([d.mean() for d in det_per_anchor]).mean()

        sigreg_g = self.sigreg(gather_embeddings(z_g))
        sigreg_t = (
            self.sigreg(gather_embeddings(z_t))
            if self.lambda_t > 0
            else torch.zeros((), device=z_g.device)
        )

        assert self.lambda_g + self.lambda_t <= 1
        total = (
            (1.0 - self.lambda_g - self.lambda_t) * align
            + self.lambda_g * sigreg_g
            + self.lambda_t * sigreg_t
        )

        with torch.no_grad():
            # Vol = sqrt(det G) in [0, 1], and the graph-text cosine, which is
            # the quantity the sign degeneracy shows up in: |cos| -> 1 with
            # mean cos -> 0 means the batch has split across the two minima.
            volume = torch.stack(
                [d.clamp_min(0.0).sqrt().mean() for d in det_per_anchor]
            ).mean()
            cos_align = (
                (
                    normalize(out['pred_g'].float(), dim=-1)
                    * normalize(z_t.float(), dim=-1)
                )
                .sum(-1)
                .mean()
            )

        return {
            'loss': total,
            # Keyed 'cross' so the existing logging in main.py is unchanged.
            'cross': align.detach(),
            'sigreg_graph': sigreg_g.detach(),
            'sigreg_text': sigreg_t.detach(),
            'volume': volume,
            'cos_align': cos_align,
        }


def build_legtjepa_loss(args: LeGTJEPAArguments) -> torch.nn.Module:
    """Select the alignment term from config: 'mse' (default) or 'volume'."""
    from tgfm.models.losses.legtjepaloss import LeGTJEPALoss

    objective = getattr(args, 'align_objective', 'mse')
    if objective == 'mse':
        return LeGTJEPALoss(args)
    if objective == 'volume':
        return LeGTJEPAVolumeLoss(args)
    raise ValueError(f'unknown align_objective {objective!r}')
