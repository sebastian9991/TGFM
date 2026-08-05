import os.path as osp
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from tgfm.utils.path import get_scratch


def get_raw_text_history(use_text: bool = False, seed: int = 0) -> Tuple[Data, List]:
    scratch = get_scratch()
    path = scratch / 'graph_clip_datasets' / 'processed' / 'history.pt'
    if osp.exists(str(path)):
        data = torch.load(path, map_location='cpu')
        # data.x = data.x.float() # Half into Float
        edge_index = to_undirected(data.edge_index)
        # edge_index, _ = add_self_loops(data.edge_index)
        data.edge_index = edge_index
        data.num_nodes = data.y.shape[0]

        # split data
        node_id = np.arange(data.num_nodes)
        np.random.shuffle(node_id)

        data.train_id = np.sort(node_id[: int(data.num_nodes * 0.6)])
        data.val_id = np.sort(
            node_id[int(data.num_nodes * 0.6) : int(data.num_nodes * 0.8)]
        )
        data.test_id = np.sort(node_id[int(data.num_nodes * 0.8) :])

        data.train_mask = torch.tensor(
            [x in data.train_id for x in range(data.num_nodes)]
        )
        data.val_mask = torch.tensor([x in data.val_id for x in range(data.num_nodes)])
        data.test_mask = torch.tensor(
            [x in data.test_id for x in range(data.num_nodes)]
        )
        return data, data.raw_texts
    else:
        raise NotImplementedError('No existing history dataset!')
