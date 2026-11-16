"""Symmetric InfoNCE summed over every modality pair (GraphCLIP-style, k >= 2).

    L = sum_{a < b} (CE(s U_a U_b^T, I) + CE(s U_b U_a^T, I)) / 2,
    U_m = unit(Z_m) over the all-gathered global batch,  s = exp(logit_scale)

At k = 2 this is GraphCLIP's objective (one pair). At k = 3 it is the
contrastive counterpart of the Gramian volume: C(3, 2) = 3 pairs, each with its
own B x B logit matrix, which is the O(C(k,2) B^2 d) cost the paper compares
against. No SIGReg: contrastive negatives are the anti-collapse term.

Selected with align_objective='infonce' on the LeGTJEPA model, so the encoder,
projections and data are identical to the volume and pairwise-MSE arms and only
the objective differs. The predictors are not used by this loss; mm_main.py
freezes them so DDP does not wait on gradients that never arrive.

Under DDP every rank all-gathers each modality (with autograd) and computes the
full global logits, as GraphCLIPLoss does; DDP's gradient averaging then gives
dL/dtheta of the single global loss.
"""

import math
from itertools import combinations
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy, normalize

from tgfm.models.losses.volumeloss import gather_embeddings
from tgfm.utils.args import LeGTJEPAArguments


class MultiModalInfoNCELoss(torch.nn.Module):
    # CLIP clamps the learnable temperature at 100.
    MAX_LOGIT_SCALE = math.log(100.0)

    def __init__(self, args: LeGTJEPAArguments) -> None:
        super().__init__()
        self.use_text = getattr(args, 'use_text', True)
        self.use_image = getattr(args, 'use_image', False)
        # Learnable temperature, initialised as in GraphCLIP / CLIP. It is a
        # parameter of the loss, so the optimizer must include
        # criterion.parameters().
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def _modalities(self, out: Dict[str, Tensor]) -> List[Tuple[str, Tensor]]:
        mods = [('g', out['z_g'])]
        if self.use_text:
            mods.append(('t', out['z_t']))
        if self.use_image:
            mods.append(('i', out['z_image']))
        return mods

    def forward(self, out: Dict[str, Tensor]) -> Dict[str, Tensor]:
        units = {
            name: normalize(gather_embeddings(z), dim=-1)
            for name, z in self._modalities(out)
        }
        scale = self.logit_scale.clamp(max=self.MAX_LOGIT_SCALE).exp()
        n = next(iter(units.values())).size(0)
        target = torch.arange(n, device=out['z_g'].device)

        total = torch.zeros((), device=target.device)
        logs: Dict[str, Tensor] = {}
        for a, b in combinations(units, 2):
            logits = scale * units[a] @ units[b].T  # (B, B)
            pair = 0.5 * (
                cross_entropy(logits, target) + cross_entropy(logits.T, target)
            )
            total = total + pair
            logs[f'infonce_{a}{b}'] = pair.detach()
        return {
            'loss': total,
            'infonce': total.detach(),
            'logit_scale': scale.detach(),
        } | logs
