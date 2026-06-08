from typing import Any, Iterator, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric import datasets
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import BaseTransform, NormalizeFeatures
from torch_geometric.utils import to_undirected

from tgfm.views.prepare import PreparedSubgraph, compute_rwse, prepare_subgraph


def get_dataset(
    root: str, name: str, transform: BaseTransform = NormalizeFeatures()
) -> InMemoryDataset:
    pyg_dataset_dict = {
        'coauthor-cs': (datasets.Coauthor, {'name': 'CS'}),
        'coauthor-physics': (datasets.Coauthor, {'name': 'physics'}),
        'amazon-computers': (datasets.Amazon, {'name': 'Computers'}),
        'amazon-photos': (datasets.Amazon, {'name': 'Photo'}),
        'cora': (datasets.Planetoid, {'name': 'Cora'}),
        'proteins': (datasets.TUDataset, {'name': 'PROTEINS'}),
        'wikics': (datasets.WikiCS, {}),
    }

    if name not in pyg_dataset_dict:
        raise ValueError(
            f'Unknown dataset {name}, Available: {list(pyg_dataset_dict.keys())}'
        )

    dataset_class, kwargs = pyg_dataset_dict[name]
    if transform is None:
        transform = None if name == 'wikics' else NormalizeFeatures()

    dataset = dataset_class(root, transform=transform, **kwargs)

    return dataset


def get_wiki_cs(root: str, transform: BaseTransform = NormalizeFeatures()) -> Any:
    dataset = datasets.WikiCS(root, transform=transform)
    data = dataset[0]
    std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
    data.x = (data.x - mean) / std
    data.edge_index = to_undirected(data.edge_index)
    return (
        [data],
        np.array(data.train_mask),
        np.array(data.val_mask),
        np.array(data.test_mask),
    )


class FullGraphEncodingsCache:
    """Caches PE/SE computed once on the full graph and slices them per ego-net.

    Usage:
        cache = FullGraphEncodingsCache(full_data, K=16)
        cache.precompute()              # runs once
        ego = next(iter(loader))        # ego-net from NeighborLoader
        ego_rwse = cache.slice(ego)     # (ego.num_nodes, K)
    """

    def __init__(self, full_data: Data, K: int = 16, se: Optional[Tensor] = None):
        self.full_data = full_data
        self.K = K
        self.full_rwse: Optional[Tensor] = None
        self.full_se = se  # caller may attach precomputed SE here

    def precompute(self) -> None:
        if self.full_rwse is None:
            self.full_rwse = compute_rwse(self.full_data, K=self.K)

    def slice(self, ego: Data) -> Tuple[Tensor, Optional[Tensor]]:
        """Slice cached encodings down to the ego-net's nodes.

        NeighborLoader attaches `n_id` to the sampled batch: a (ego.num_nodes,)
        LongTensor mapping ego-net local indices to full-graph node ids.

        Returns (rwse_ego, se_ego). If SE wasn't precomputed, se_ego is None.
        """
        if self.full_rwse is None:
            raise RuntimeError('Call precompute() before slice().')
        if not hasattr(ego, 'n_id'):
            raise AttributeError(
                "Sampled batch is missing `n_id`; ensure you're using "
                'NeighborLoader (which adds it automatically).'
            )
        n_id = ego.n_id
        rwse_ego = self.full_rwse[n_id]
        se_ego = self.full_se[n_id] if self.full_se is not None else None
        return rwse_ego, se_ego


class SubgraphPreparer:
    """Wraps a NeighborLoader and produces `PreparedSubgraph` objects.

    Each iteration yields one PreparedSubgraph (one ego-net with its
    global+local views). Compose `batch_size` of these in the training loop
    to form a LeJEPA batch (B subgraphs).

    Args:
        full_data:         the (single) full-graph Data from `get_dataset`.
        cache:             pre-fitted FullGraphEncodingsCache.
        num_neighbors:     NeighborLoader sampling fan-out per hop. Default
                           [10, 10] = 2-hop ego-net with 10 neighbors per hop.
        seed_batch_size:   NeighborLoader's batch_size; how many seed nodes
                           per sampled ego-net. Setting this to 1 gives one
                           seed per ego-net (smaller, faster); larger values
                           give bigger ego-nets that share neighborhood
                           expansion (faster amortized, but coarser).
        prepare_kwargs:    forwarded to `prepare_subgraph` (num_local_parts,
                           num_global_views, etc.).
        input_nodes:       restrict NeighborLoader to a subset of seeds (e.g.
                           train_mask). None -> all nodes are valid seeds,
                           which is appropriate for SSL.
    """

    def __init__(
        self,
        full_data: Data,
        cache: FullGraphEncodingsCache,
        num_neighbors: list[int] | None = None,
        seed_batch_size: int = 1,
        prepare_kwargs: Optional[dict] = None,
        input_nodes: Optional[Tensor] = None,
        shuffle: bool = True,
        num_workers: int = 0,
    ):
        self.full_data = full_data
        self.cache = cache
        self.prepare_kwargs = prepare_kwargs or {}
        self.num_neighbors = num_neighbors or [10, 10]
        self.seed_batch_size = seed_batch_size

        self.loader = NeighborLoader(
            full_data,
            num_neighbors=self.num_neighbors,
            batch_size=seed_batch_size,
            input_nodes=input_nodes,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        self._it: Optional[Iterator[Data]] = None

    def _next_ego(self) -> Data:
        if self._it is None:
            self._it = iter(self.loader)
        try:
            ego = next(self._it)
            # ego.edge_index = to_undirected(ego.edge_index, num_nodes=ego.num_nodes)
            return ego
        except StopIteration:
            self._it = iter(self.loader)
            ego = next(self._it)
            # ego.edge_index = to_undirected(ego.edge_index, num_nodes=ego.num_nodes)
            return ego

    # def __iter__(self) -> Iterator[PreparedSubgraph]:
    #     for ego in self.loader:
    #         rwse_ego, se_ego = self.cache.slice(ego)
    #         prep = prepare_subgraph(
    #             ego,
    #             rwse=rwse_ego,
    #             se=se_ego,
    #             **self.prepare_kwargs,
    #         )
    #         assert is_undirected(prep.source.edge_index) == True
    #         yield prep

    def sample_batch(self, batch_size: int) -> list[PreparedSubgraph]:
        """Convenience: pull `batch_size` PreparedSubgraphs in sequence.

        Wraps __iter__ and stops after batch_size yields. Re-creates the
        iterator on each call (so you get fresh samples each step).
        """
        out: list[PreparedSubgraph] = []
        for _ in range(batch_size):
            ego = self._next_ego()
            rwse_ego, se_ego = self.cache.slice(ego)
            prep = prepare_subgraph(
                ego,
                rwse=rwse_ego,
                se=se_ego,
                **self.prepare_kwargs,
            )
            out.append(prep)
            # assert is_undirected(prep.source.edge_index) == True
        return out
