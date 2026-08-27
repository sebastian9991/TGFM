"""LeGTJEPA objective with a Gramian volume alignment term (graph/text/image).
    L = (1 - sum_a lambda_a) * L_vol  +  sum_a lambda_a * SIGReg(Z^(a))
    L_vol = mean_a mean_i det G_i^(a)
For sample i and anchor modality a, the columns of A_i^(a) are the k modality
directions, with the anchor column replaced by that modality's prediction and
the other k-1 columns detached:
    A_i^(a) = [ unit(h_a(z_i^(a))) | sg(unit(z_i^(b))) for b != a ]   (d, k)
    G_i^(a) = A_i^(a).T @ A_i^(a)                                     (k, k)
det G_i^(a) is the squared volume of the k-parallelotope spanned by those
columns (Gantmacher 1959; Thm. 1 of Cicchetti et al., GRAM, ICLR 2025). The
columns are unit norm, so diag(G) = 1 and det G in [0, 1]: 1 when the
directions are mutually orthogonal, 0 when they are collinear.
The k = 3 expansion is
    det G = 1 - c_gt^2 - c_gi^2 - c_ti^2 + 2 c_gt c_gi c_ti  (GRAM Eq. 13),
where the triple product 2 c_gt c_gi c_ti is the joint term with no pairwise
equivalent -- the reason the three-modality volume is more than a sum of the
three pairwise MSE losses.
The three rows (z_g, z_t, z_image) must be aligned per sample: the subgraph
around node v, v's text, v's image -- one positive triple per center node.
Under DDP, embeddings are gathered across ranks before SIGReg so the
regularizer sees the global batch marginal (LeVLJEPA App. A); the per-sample
volume term needs no gathering.
Set align_objective='volume' and use_image=True in the config to select this.
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
    """Det G for a batch of k x k Gram matrices; (..., k, k) -> (...).
    Cofactor expansion along the first row, written out for the k = 2 and
    k = 3 cases; larger k defers to torch.linalg.det.
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


def squared_volume(
    embeddings: List[Tensor], predictions: List[Tensor], anchor: int
) -> Tensor:
    """Det G_i^(a) for every sample in the batch, at a fixed anchor a.
    embeddings[b] and predictions[b] are (B, d) for modality b. Only
    predictions[anchor] carries gradient; every other column is detached, so
    gradients reach each encoder solely through its own branch.
    """
    columns = [
        normalize(
            predictions[anchor] if b == anchor else z_b.detach(), dim=-1, eps=1e-8
        )
        for b, z_b in enumerate(embeddings)
    ]
    A = torch.stack(columns, dim=-1)  # (B, d, k)
    G = A.transpose(-2, -1) @ A  # (B, k, k), diag(G) = 1
    return gramian(G)  # (B,)


class LeGTJEPAVolumeLoss(torch.nn.Module):
    def __init__(self, args: LeGTJEPAArguments) -> None:
        super().__init__()
        self.use_image = getattr(args, 'use_image', False)
        self.lambda_g = args.lambda_graph
        # A branch with no trainable parameters upstream contributes no
        # gradient: drop its anchor term and its SIGReg term. Text is frozen
        # via freeze_text_projection; image has no frozen-projection flag, so
        # its branch is active whenever use_image is set.
        self.lambda_t = 0.0 if args.freeze_text_projection else args.lambda_text
        self.text_branch_active = not args.freeze_text_projection
        self.lambda_i = getattr(args, 'lambda_image', 0.0) if self.use_image else 0.0

        self.sigreg = SIGReg(
            num_slices=args.sigreg_num_slices,
            num_quad_points=args.sigreg_num_quad_points,
            t_max=args.sigreg_t_max,
        )

    def forward(self, out: Dict[str, Tensor]) -> Dict[str, Tensor]:
        z_g, z_t = out['z_g'], out['z_t']
        if self.use_image:
            embeddings = [z_g, z_t, out['z_image']]
            predictions = [out['pred_g'], out['pred_t'], out['pred_image']]
            anchors = [0]  # graph always active
            if self.text_branch_active:
                anchors.append(1)
            anchors.append(2)  # image active whenever use_image
        else:
            embeddings = [z_g, z_t]
            predictions = [out['pred_g'], out['pred_t']]
            anchors = [0, 1] if self.text_branch_active else [0]

        dets = [squared_volume(embeddings, predictions, a) for a in anchors]
        # det G is per-sample in [0, 1]: mean over batch, then over anchors.
        cross = torch.stack([d.mean() for d in dets]).mean()

        sigreg_g = self.sigreg(gather_embeddings(z_g))
        sigreg_t = (
            self.sigreg(gather_embeddings(z_t))
            if self.lambda_t > 0
            else torch.zeros((), device=z_g.device)
        )
        sigreg_i = (
            self.sigreg(gather_embeddings(out['z_image']))
            if self.lambda_i > 0
            else torch.zeros((), device=z_g.device)
        )

        lambda_tot = self.lambda_g + self.lambda_t + self.lambda_i
        assert lambda_tot <= 1
        total = (
            (1.0 - lambda_tot) * cross
            + self.lambda_g * sigreg_g
            + self.lambda_t * sigreg_t
            + self.lambda_i * sigreg_i
        )

        logs = {
            'loss': total,
            'cross': cross.detach(),
            'sigreg_graph': sigreg_g.detach(),
            'sigreg_text': sigreg_t.detach(),
        }
        with torch.no_grad():
            logs['volume'] = torch.stack(
                [d.clamp_min(0.0).sqrt().mean() for d in dets]
            ).mean()
            # Pairwise cosines are the sign / degeneracy diagnostics: any |c|
            # near 1 with mean c near 0 means a batch split across the two
            # even-function minima; det G collapsing while one pair stays
            # orthogonal means an ignored modality (adj(G)=0 kills its grad).
            zg = normalize(z_g, dim=-1)
            zt = normalize(z_t, dim=-1)
            logs['cos_gt'] = (zg * zt).sum(-1).mean()
            if self.use_image:
                logs['sigreg_image'] = sigreg_i.detach()
                zi = normalize(out['z_image'], dim=-1)
                logs['cos_gi'] = (zg * zi).sum(-1).mean()
                logs['cos_ti'] = (zt * zi).sum(-1).mean()
        return logs
