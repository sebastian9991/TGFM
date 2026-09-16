"""GraphCLIP objective for MM-Graph pretraining (G-T / G-I).

Symmetric InfoNCE, computed by GraphCLIP's own functions (models/dp.py):

    logits = s * unit(Z_g) unit(Z_p)^T,   s = exp(logit_scale)
    L = (CE(logits, I) + CE(logits^T, I)) / 2

train.py runs torch DataParallel, which gathers every replica's embeddings onto
one device, so the softmax sees the whole global batch (7200). Under DDP the
same global logits need an explicit gather: embeddings are all-gathered with
autograd (the gather's backward returns each rank its share of the gradient),
and DDP's gradient averaging then recovers dL/dtheta of the single global loss.
"""

from typing import Dict

import torch
from torch import Tensor

from tgfm.evaluation.graphclip_mm_adapter import graphclip_on_path
from tgfm.models.losses.volumeloss import gather_embeddings


class GraphCLIPLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        graphclip_on_path()
        from models.dp import calculate_loss, create_logits

        self._create_logits = create_logits
        self._calculate_loss = calculate_loss
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, out: Dict[str, Tensor]) -> Dict[str, Tensor]:
        z_g = gather_embeddings(out['z_g'])
        z_p = gather_embeddings(out['z_p'])
        scale = out['logit_scale'].exp()

        logits_g, logits_p = self._create_logits(z_g, z_p, scale)
        loss = self._calculate_loss(logits_g, logits_p, self.criterion)

        with torch.no_grad():
            target = torch.arange(logits_g.size(0), device=logits_g.device)
            g2p = (logits_g.argmax(1) == target).float().mean()
            p2g = (logits_p.argmax(1) == target).float().mean()
        return {
            'loss': loss,
            'infonce': loss.detach(),
            'logit_scale': scale.detach(),
            'r1_g2p': g2p,  # in-batch retrieval R@1, collapse diagnostic
            'r1_p2g': p2g,
        }
