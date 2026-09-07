"""Lazy ego-subgraph sampler for MM-Graph.

Replaces parse_target_data for the MM-Graph datasets. That function reads a
precomputed JSON of ego-subgraphs (graph_clip_datasets/target_data/<name>.json)
produced by GraphCLIP's preprocessing; MM-Graph ships no such file, so the
subgraphs are built here directly from the edge index.

The per-subgraph contract is identical to parse_target_data, so the graph
encoder sees the same inputs:
    edge_index   reindexed to the local node set
    x            data.x[node_idx]              (text features)
    root_n_index position of the center node in the local indexing
    pe           RWPE-32, added per subgraph by AddRandomWalkPE
    y            data.y[u]                     (NC datasets only)
plus one addition for the image tower:
    image_x      data.image_x[u], shape (1, image_dim)

Two differences from parse_target_data, both forced by scale:

1. LAZY. Subgraphs are built in __getitem__ rather than materialized up front.
   books-nc is 685,294 nodes; holding that many Data objects (each with its own
   edge_index, feature block and 32-d PE) exhausts memory long before training
   starts. As a Dataset this streams through the DataLoader instead, and
   indexing is unchanged for callers doing graphs[i].

2. SAMPLED NEIGHBOURHOODS. num_hops/fanout cap the ego-subgraph. books-nc has
   average degree 21, so an uncapped 2-hop neighbourhood averages ~440 nodes
   and the tail is far worse; the fanout keeps subgraph size bounded and the
   RWPE cost per sample flat.

Node ordering is the identity: item u is the ego-subgraph of node u. The image
and label rows are indexed by u directly, so this ordering is what keeps the
three modalities row-aligned in the volume term. assert_alignment() checks it.
"""

from typing import Optional

import torch
import torch_geometric.transforms as T
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import sort_edge_index


class MMEgoDataset(Dataset):
    """One ego-subgraph per node, built on demand."""

    def __init__(
        self,
        data: Data,
        num_hops: int = 2,
        fanout: int = 12,
        walk_length: int = 32,
        seed: int = 0,
    ) -> None:
        self.num_nodes = int(data.x.size(0))
        self.x = data.x
        self.image_x = getattr(data, 'image_x', None)
        self.y = getattr(data, 'y', None)
        self.num_hops = num_hops
        self.fanout = fanout
        self.transform = T.AddRandomWalkPE(walk_length=walk_length, attr_name='pe')
        self._generator = torch.Generator().manual_seed(seed)

        # CSR adjacency for O(deg) neighbour lookup. Built once per dataset.
        edge_index = sort_edge_index(data.edge_index, num_nodes=self.num_nodes)
        self.col = edge_index[1].contiguous()
        deg = torch.bincount(edge_index[0], minlength=self.num_nodes)
        self.rowptr = torch.cat(
            [torch.zeros(1, dtype=torch.long), deg.cumsum(0)]
        ).contiguous()

    def __len__(self) -> int:
        return self.num_nodes

    def _neighbours(self, nodes: Tensor) -> Tensor:
        """Sampled out-neighbours of a set of nodes, capped at fanout each."""
        out = []
        for n in nodes.tolist():
            lo, hi = int(self.rowptr[n]), int(self.rowptr[n + 1])
            if hi <= lo:
                continue
            nbrs = self.col[lo:hi]
            if nbrs.numel() > self.fanout:
                perm = torch.randperm(nbrs.numel(), generator=self._generator)
                nbrs = nbrs[perm[: self.fanout]]
            out.append(nbrs)
        if not out:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(out)

    def __getitem__(self, u: int) -> Data:
        u = int(u)
        frontier = torch.tensor([u], dtype=torch.long)
        nodes = {u}
        for _ in range(self.num_hops):
            nbrs = self._neighbours(frontier)
            new = torch.tensor(
                sorted(set(nbrs.tolist()) - nodes), dtype=torch.long
            )
            if new.numel() == 0:
                break
            nodes.update(new.tolist())
            frontier = new

        node_idx = torch.tensor(sorted(nodes), dtype=torch.long)
        local = {int(g): i for i, g in enumerate(node_idx.tolist())}

        # Induced edges among the sampled node set, reindexed locally.
        src, dst = [], []
        node_set = set(node_idx.tolist())
        for g in node_idx.tolist():
            lo, hi = int(self.rowptr[g]), int(self.rowptr[g + 1])
            for v in self.col[lo:hi].tolist():
                if v in node_set:
                    src.append(local[g])
                    dst.append(local[v])
        if not src:
            # Isolated node: self-loop, matching parse_target_data's fallback.
            edge_index = torch.tensor([[local[u]], [local[u]]], dtype=torch.long)
        else:
            edge_index = torch.tensor([src, dst], dtype=torch.long)

        graph = Data(
            edge_index=edge_index,
            x=self.x[node_idx],
            root_n_index=local[u],
        )
        if self.y is not None:
            graph.y = self.y[u].view(1)
        if self.image_x is not None:
            graph.image_x = self.image_x[u].unsqueeze(0)  # (1, image_dim)
        graph.num_nodes = node_idx.numel()
        return self.transform(graph)

    def assert_alignment(self, n_check: int = 8) -> None:
        """Verify item u really is node u's ego-subgraph, for the first n."""
        for u in range(min(n_check, self.num_nodes)):
            g = self[u]
            assert torch.allclose(g.x[g.root_n_index], self.x[u]), (
                f'subgraph {u}: root feature does not match node {u}'
            )
            if self.image_x is not None:
                assert torch.allclose(g.image_x[0], self.image_x[u])
            if self.y is not None:
                assert int(g.y) == int(self.y[u])


def parse_mm_target_data(
    name: str,
    data: Data,
    num_hops: int = 2,
    fanout: int = 12,
    walk_length: int = 32,
) -> MMEgoDataset:
    """MM-Graph replacement for parse_target_data; indexable like a list."""
    return MMEgoDataset(
        data, num_hops=num_hops, fanout=fanout, walk_length=walk_length
    )
