"""Data preparation: RWSE, METIS local patches, BFS/random-walk global views.

This file is intentionally a stub-with-skeletons. It compiles & runs end-to-end
on small synthetic graphs, but the production-grade implementations of METIS
partitioning, BFS expansion, and RWSE should be cross-checked against:
    - torch_geometric.transforms.AddRandomWalkPE (for RWSE)
    - torch_geometric.utils.subgraph + cluster_data / ClusterData (for METIS)
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData  # METIS wrapper
from torch_geometric.transforms import AddRandomWalkPE
from torch_geometric.utils import k_hop_subgraph, subgraph, to_undirected


@dataclass
class PreparedSubgraph:
    """All views needed by one LeJEPA training step for one source subgraph."""

    source: Data  # the original subgraph (post-NeighborLoader)
    rwse: Tensor  # (N, K) node-level RWSE for the source
    global_views: List[Data] = field(default_factory=list)
    local_views: List[Data] = field(default_factory=list)


def safe_cluster(data: Data, num_parts: int) -> None:
    n = data.num_nodes

    if n == 0:
        raise ValueError('METIS called on empty subgraph (0 vertices)')


def compute_rwse(data: Data, K: int) -> Tensor:
    """Random Walk Structural Encoding of walk length K.

    rwse[i, k] = Pr(i -> i in k steps of a uniform random walk on G).

    Uses torch-geometric's AddRandomWalkPE under the hood for correctness.
    """
    transform = AddRandomWalkPE(walk_length=K, attr_name='_rwse')
    out = transform(data.clone())
    return out._rwse  # (N, K)


def build_local_views_metis(
    data: Data,
    rwse: Tensor,
    num_parts: int,
    expand_hops: int = 1,
    se: Optional[Tensor] = None,
    directed: bool = False,
) -> List[Data]:
    """Partition `data` with METIS into `num_parts`, then 1-hop expand each part.

    Each returned Data carries x, edge_index, pe (=rwse[V]), and optionally
    se (=se[V]) restricted to the local view's node set.
    """
    # ClusterData internally uses METIS (via torch-sparse / pyg-lib).
    safe_cluster(data=data, num_parts=num_parts)
    N = data.num_nodes
    # METIS requires an undirected graph avoids segumentation faults.
    undirected_ei = to_undirected(data.edge_index, num_nodes=N)
    metis_data = Data(edge_index=undirected_ei, num_nodes=N)
    cluster_data = ClusterData(metis_data, num_parts=num_parts, log=False)

    views: List[Data] = []
    N = data.num_nodes
    device = data.edge_index.device

    part = getattr(cluster_data, 'partition', None)

    # ClusterData stores a permutation + partition pointers. We reconstruct
    # the original node-id sets per partition.
    assert part is not None

    perm = part.node_perm  # (N,)
    partptr = part.partptr  # (num_parts +1, )

    for p in range(num_parts):
        start, end = int(partptr[p]), int(partptr[p + 1])
        if end <= start:
            continue
        part_nodes = perm[start:end].to(device)  # original node ids in this part

        # k_hop_subgraph expands the seed set by `num_hops`.
        sub_nodes, _, _, _ = k_hop_subgraph(
            part_nodes,
            num_hops=expand_hops,
            edge_index=undirected_ei,
            relabel_nodes=False,
            num_nodes=N,
        )

        view_ei = data.edge_index if directed else undirected_ei
        sub_ei, _ = subgraph(sub_nodes, view_ei, relabel_nodes=True, num_nodes=N)
        view = Data(
            x=data.x[sub_nodes],
            edge_index=sub_ei,
            pe=rwse[sub_nodes],
        )
        view.n_id = sub_nodes
        if se is not None:
            view.se = se[sub_nodes]
        if getattr(data, 'edge_attr', None) is not None:
            # Edge attrs need to follow the kept-edge mask; for now we drop them
            # and rely on the GINEConv default-zero edge_attr path.
            # TODO: thread through edge_attr via subgraph(..., return_edge_mask=True).
            pass
        views.append(view)

    return views


def build_global_views_from_local_union(
    data: Data,
    rwse: Tensor,
    local_views: List[Data],
    num_views: int = 1,
    local_frac: float = 0.5,
    rng: Optional[torch.Generator] = None,
    se: Optional[Tensor] = None,
    directed: bool = False,
) -> List[Data]:
    """Build each global view from the union of a random subset of local views.

    Instead of sampling the source graph directly (BFS/RWR), each global view is
    assembled from the local patches already produced by
    `build_local_views_metis`: we sample ``ceil(local_frac * len(local_views))``
    of them, union their *original* node ids, and rebuild one connected-ish?
    subgraph on `data` from that union. This ties the global view's support to
    the local partition geometry.

    Each entry of `local_views` must carry `n_id` (original node ids into `data`),
    which `build_local_views_metis` attaches automatically.

    Args:
        data:        the source subgraph the local views were cut from.
        rwse:        RWSE for `data`'s nodes, sliced per view (same convention as
                     the other builders).
        local_views: the list returned by `build_local_views_metis`.
        num_views:   how many global views to produce.
        local_frac:  fraction of available local views to merge into each global
                     view. ``1.0`` merges all of them (deterministic, identical
                     every call); use ``< 1.0`` to get diverse global views.
        rng:         generator for reproducible sampling.
        se:          optional structural encoding, sliced per view.
        directed:    if True, keep `data`'s edge directions in the view; if False
                     (default, matching the other builders here) symmetrize.
    """
    if not local_views:
        return []
    for lv in local_views:
        if not hasattr(lv, 'n_id'):
            raise AttributeError(
                'local view is missing `n_id`; build it with '
                'build_local_views_metis (which attaches original node ids).'
            )

    L = len(local_views)
    k = max(1, min(L, int(round(local_frac * L))))
    N = data.num_nodes
    device = data.edge_index.device

    views: List[Data] = []
    for _ in range(num_views):
        chosen = torch.randperm(L, generator=rng)[:k].tolist()
        node_ids = torch.cat([local_views[i].n_id for i in chosen])
        nodes = torch.unique(node_ids).to(device)  # sorted, de-duplicated

        sub_ei, _ = subgraph(nodes, data.edge_index, relabel_nodes=True, num_nodes=N)
        if not directed:
            sub_ei = to_undirected(edge_index=sub_ei)

        view = Data(
            x=data.x[nodes],
            edge_index=sub_ei,
            pe=rwse[nodes],
        )
        if se is not None:
            view.se = se[nodes]
        view.n_id = nodes
        views.append(view)
    return views


def build_global_views(
    data: Data,
    rwse: Tensor,
    num_views: int = 2,
    coverage_frac: float = 0.7,
    strategy: str = 'bfs',  # "bfs" | "rwr"
    rng: Optional[torch.Generator] = None,
    se: Optional[Tensor] = None,
) -> List[Data]:
    """Sample `num_views` large connected subsamples from `data`.

    Each covers ~`coverage_frac` of the nodes. The seed and (for RWR) walks
    use `rng` for reproducibility; default is the global PyTorch generator.
    """
    N = data.num_nodes
    target_size = max(1, int(coverage_frac * N))
    views: List[Data] = []

    for _ in range(num_views):
        seed_idx = torch.randint(0, N, (1,), generator=rng).item()
        assert isinstance(seed_idx, int)
        if strategy == 'bfs':
            nodes = _bfs_expand(data.edge_index, seed_idx, target_size, N)
        elif strategy == 'rwr':
            nodes = _random_walk_restart(
                data.edge_index, seed_idx, target_size, N, rng=rng
            )
        else:
            raise ValueError(f'Unknown global view strategy: {strategy}')

        sub_ei, _ = subgraph(nodes, data.edge_index, relabel_nodes=True, num_nodes=N)
        view = Data(
            x=data.x[nodes],
            edge_index=sub_ei,
            pe=rwse[nodes],
        )
        if se is not None:
            view.se = se[nodes]
        views.append(view)
    return views


def _bfs_expand(edge_index: Tensor, seed: int, target_size: int, N: int) -> Tensor:
    """Plain BFS from `seed` until we've collected `target_size` nodes (or run out)."""
    # Build adjacency on the fly for clarity. For large graphs, precompute an
    # adjacency list once per subgraph and reuse.
    adj: list[list[int]] = [[] for _ in range(N)]
    src, dst = edge_index
    for s, d in zip(src.tolist(), dst.tolist()):
        adj[s].append(d)

    visited = {seed}
    frontier = [seed]
    order = [seed]
    while frontier and len(order) < target_size:
        next_frontier = []
        for u in frontier:
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    order.append(v)
                    next_frontier.append(v)
                    if len(order) >= target_size:
                        break
            if len(order) >= target_size:
                break
        frontier = next_frontier
    return torch.tensor(order, dtype=torch.long, device=edge_index.device)


def _random_walk_restart(
    edge_index: Tensor,
    seed: int,
    target_size: int,
    N: int,
    restart_prob: float = 0.15,
    max_steps: int = 10_000,
    rng: Optional[torch.Generator] = None,
) -> Tensor:
    """Random walk with restart from `seed`, collecting unique visited nodes."""
    adj: list[list[int]] = [[] for _ in range(N)]
    src, dst = edge_index
    for s, d in zip(src.tolist(), dst.tolist()):
        adj[s].append(d)

    visited = {seed}
    current = seed
    for _ in range(max_steps):
        if len(visited) >= target_size:
            break
        # Restart?
        if torch.rand((), generator=rng).item() < restart_prob or not adj[current]:
            current = seed
            continue
        nbrs = adj[current]
        idx = torch.randint(0, len(nbrs), (1,), generator=rng).item()
        assert isinstance(idx, int)
        current = nbrs[idx]
        visited.add(current)
    return torch.tensor(sorted(visited), dtype=torch.long, device=edge_index.device)


def prepare_subgraph(
    data: Data,
    K: int = 16,
    num_local_parts: int = 8,
    num_global_views: int = 2,
    global_coverage_frac: float = 0.7,
    global_strategy: str = 'bfs',
    global_local_frac: float = 0.5,
    num_local_as_global: int = 0,
    rwse: Optional[Tensor] = None,
    se: Optional[Tensor] = None,
) -> PreparedSubgraph:
    """Prepare global+local views for one subgraph.

    Args:
        data: PyG `Data` representing the subgraph to view (post-NeighborLoader
              ego-net, or post-ClusterLoader partition, or a small whole graph).
        K:    RWSE walk length (used only if `rwse` is None).
        num_local_parts: The number of local views.
        num_global_views: The number of global views.
        global_coverage_frac: Coverage of subgraph nodes to use, defines global view size.
        global_strategy: Strategy of constructing the global view.
        global_local_frac: Fraction of coverage in union of local node ids for global view.
        num_local_as_global: Integer which defines the slices of local views to take as global view.
        rwse: Optional precomputed RWSE restricted to `data`'s nodes (shape
              (N, K)). For transductive node datasets you usually want to
              compute RWSE *once on the full graph* and slice it down to the
              sampled ego-net's `n_id` (see `RWSECache.slice_for`) — computing
              RWSE on the ego-net itself measures self-returns on the truncated
              subgraph, which is *not* the same statistic.
        se:   Optional precomputed structural encoding (any shape (N, S)).
              Stored on every view's Data as `data.se` for the encoder's SE
              branch. Same slicing caveat as `rwse`.
    """
    if rwse is None:
        rwse = compute_rwse(data, K=K)

    locals_ = build_local_views_metis(
        data,
        rwse,
        num_parts=num_local_parts,
        expand_hops=1,
        se=se,
    )
    if global_strategy == 'local':
        if num_local_as_global > 0:
            k = min(num_local_as_global, len(locals_))
            if k == len(locals_):
                globals_ = locals_[:1]
                locals_ = locals_[1:]
            else:
                globals_ = locals_[:k]
                locals_ = locals_[k:]
        else:
            globals_ = build_global_views_from_local_union(
                data,
                rwse,
                local_views=locals_,
                num_views=num_global_views,
                local_frac=global_local_frac,
                se=se,
            )
    else:
        globals_ = build_global_views(
            data,
            rwse,
            num_views=num_global_views,
            coverage_frac=global_coverage_frac,
            strategy=global_strategy,
            se=se,
        )
    return PreparedSubgraph(
        source=data, rwse=rwse, global_views=globals_, local_views=locals_
    )
