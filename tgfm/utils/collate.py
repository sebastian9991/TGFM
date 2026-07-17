from copy import deepcopy
from typing import List, Optional, Tuple

import dgl
import torch
from dgl import transforms as T
from dgl.transforms import BaseTransform
from torch import Tensor

from tgfm.utils.args import ModelArguments, TransferArguments


class Universal_Collator(object):
    """Trimmed to the two-augmentation (GRACE-style) branch used by SIGReg.

    Kept consistent with the original GSTBench collator: 'sigreg' uses the same
    feature-mask + edge-drop transforms as 'grace'.
    """

    def __init__(self, task: str, args: ModelArguments, device: torch.device) -> None:
        self.device = device
        self.task = task.lower()
        assert isinstance(args, TransferArguments)
        self.transforms: Optional[BaseTransform] = None

        if self.task in ('grace', 'sigreg', 'bgrl'):
            t1 = dgl.transforms.FeatMask(node_feat_names=['feat'], p=args.p_feat_drop)
            t2 = dgl.transforms.DropEdge(args.p_edge_drop)
            if args.make_undirected:
                t3 = dgl.transforms.AddReverse()
                self.transforms = T.Compose([t1, t2, t3])
            else:
                self.transforms = T.Compose([t1, t2])
        else:
            raise ValueError(f'Not implemented: {task}.')

    def __call__(self, gs: List[dgl.DGLGraph]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        assert self.transforms is not None
        g1 = deepcopy(gs[0])
        g2 = deepcopy(gs[0])
        g1 = self.transforms(g1)
        x1 = g1.ndata['feat']
        src, dst = g1.edges()
        e1 = torch.stack([src, dst], dim=1)

        g2 = self.transforms(g2)
        x2 = g2.ndata['feat']
        src, dst = g2.edges()
        e2 = torch.stack([src, dst], dim=1)
        return x1, e1, x2, e2
