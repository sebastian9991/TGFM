import os.path as osp
import random
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected

# return cora dataset as pytorch geometric Data object together with 60/20/20 split, and list of cora IDs
data_path = '../datasets'

from tgfm.utils.path import get_scratch


def get_cora_casestudy(SEED: int = 0) -> Tuple[Data, List]:
    data_X, data_Y, data_citeid, data_edges = parse_cora()
    # data_X = sklearn.preprocessing.normalize(data_X, norm="l1")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    data_name = 'cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid(data_path, data_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    data.x = torch.tensor(data_X).float()
    data.edge_index = torch.tensor(data_edges).long()
    data.y = torch.tensor(data_Y).long()
    data.num_nodes = len(data_Y)

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[: int(data.num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(data.num_nodes * 0.6) : int(data.num_nodes * 0.8)]
    )
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8) :])

    data.train_mask = torch.tensor([x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor([x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor([x in data.test_id for x in range(data.num_nodes)])

    return data, data_citeid


# credit: https://github.com/tkipf/pygcn/issues/27, xuhaiyun


def parse_cora() -> Tuple[npt.NDArray, npt.NDArray, List, npt.NDArray]:
    path = f'{data_path}/cora_orig/cora'
    idx_features_labels = np.genfromtxt('{}.content'.format(path), dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {
        x: i
        for i, x in enumerate(
            [
                'Case_Based',
                'Genetic_Algorithms',
                'Neural_Networks',
                'Probabilistic_Methods',
                'Reinforcement_Learning',
                'Rule_Learning',
                'Theory',
            ]
        )
    }
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt('{}.cites'.format(path), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape
    )
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()


def get_raw_text_cora(
    use_text: bool = False, seed: int = 0
) -> Tuple[Data, Optional[List]]:
    scratch = get_scratch()
    path = scratch / 'graph_clip_datsets' / 'processed' / 'cora.pt'
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
    data, data_citeid = get_cora_casestudy(seed)
    if not use_text:
        return data, None

    with open(f'{data_path}/cora_orig/mccallum/cora/papers') as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1]
        pid_filename[pid] = fn

    path = f'{data_path}/cora_orig/mccallum/cora/extractions/'
    text = []
    for pid in data_citeid:
        fn = pid_filename[pid]
        with open(path + fn) as f:
            lines = f.read().splitlines()

        for line in lines:
            if 'Title:' in line:
                ti = line
            if 'Abstract:' in line:
                ab = line
        text.append(ti + '\n' + ab)
    return data, text
