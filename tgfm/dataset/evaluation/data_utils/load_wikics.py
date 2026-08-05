import os.path as osp
from typing import List, Tuple

import torch
from torch_geometric.data import Data

from tgfm.utils.path import get_scratch


def get_raw_text_wikics(use_text: bool = False, seed: int = 0) -> Tuple[Data, List]:
    scratch = get_scratch()
    path = scratch / 'graph_clip_datasets' / 'processed' / 'wikics.pt'
    if osp.exists(str(path)):
        data = torch.load(path, map_location='cpu')
        data.train_mask = data.train_mask[:, seed]
        data.val_mask = data.val_mask[:, seed]
        # data.test_mask = data.test_masks[seed]
        return data, data.raw_texts
    else:
        raise NotImplementedError('No existing wikics dataset!')
