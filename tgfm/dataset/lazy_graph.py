from typing import List

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset


class LazyGraphDataset(Dataset):
    def __init__(self, graph_list: List[dgl.DGLGraph], features_path: str):
        """Initialize the dataset.

        :param graph_list: A list of DGL Graph objects.
        :param features_path: Path to the features tensor saved on disk.
        """
        self.graph_list = graph_list
        # Load the features tensor using memory mapping
        # self.features = torch.load(features_path, map_location=torch.device('cpu'), mmap=True)
        self.features = None
        self.features_path = features_path

    def __len__(self) -> int:
        """Return the length of the dataset (number of graphs)."""
        return len(self.graph_list)

    def __getitem__(self, idx: int) -> dgl.DGLGraph:
        """Return a single graph and its corresponding features.

        :param idx: Index of the graph in the dataset.
        """
        if self.features is None:
            # self.features = np.memmap(self.features_path, dtype='float32', mode='r')
            self.features = np.load(self.features_path, mmap_mode='r')
            # self.features = np.load(self.features_path)

        graph = self.graph_list[idx]

        # Check if 'dgl.NID' feature exists
        if dgl.NID in graph.ndata:
            assert self.features is not None
            nids = graph.ndata[dgl.NID]
            # Load the corresponding features from the features tensor
            graph.ndata['feat'] = torch.tensor(
                self.features[nids.numpy()], dtype=torch.float32
            )
            # del graph.ndata[dgl.NID]
        else:
            raise KeyError(f'{dgl.NID} feature not found in graph')

        return graph
