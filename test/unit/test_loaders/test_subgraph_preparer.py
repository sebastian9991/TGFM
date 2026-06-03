import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import stochastic_blockmodel_graph

from tgfm.dataset.pyg.data import FullGraphEncodingsCache, SubgraphPreparer
from tgfm.views.prepare import PreparedSubgraph

NUM_BLOCKS = 4
NODES_PER_BLOCK = 30  # total N = 120
P_INTRA = 0.25  # within-block edge probability
P_INTER = 0.02  # between-block edge probability
FEATURE_DIM = 16
SEED = 42

@pytest.fixture(scope="session")
def medium_graph() -> Data:
    """A 120-node SBM graph with 4 blocks of 30 nodes each.

    Connectivity: dense within blocks (p=0.25), sparse between (p=0.02).
    Resulting graph is well-connected (one giant component with high
    probability) but has clear community structure for METIS to find.

    Returns a PyG `Data` with:
        x:          (120, 16) random node features
        edge_index: (2, E) undirected edges (symmetric)
        y:          (120,) block id per node, useful for asserting that
                    METIS partitions mostly respect block boundaries
    """
    torch.manual_seed(SEED)

    block_sizes = [NODES_PER_BLOCK] * NUM_BLOCKS
    # Edge probability matrix: P_INTRA on diagonal, P_INTER off-diagonal.
    p = torch.full((NUM_BLOCKS, NUM_BLOCKS), P_INTER)
    p.fill_diagonal_(P_INTRA)

    edge_index = stochastic_blockmodel_graph(block_sizes, p, directed=False)

    N = sum(block_sizes)
    x = torch.randn(N, FEATURE_DIM)
    y = torch.cat(
        [torch.full((s,), b, dtype=torch.long) for b, s in enumerate(block_sizes)]
    )

    return Data(x=x, edge_index=edge_index, y=y, num_nodes=N)


def test_cache_precompute_shape(medium_graph):
    """precompute() yields (N, K) RWSE matching the cache's K parameter."""
    K = 16
    cache = FullGraphEncodingsCache(full_data=medium_graph, K=K)
    cache.precompute()
    assert cache.full_rwse is not None
    assert cache.full_rwse.shape == (medium_graph.num_nodes, K)
    # RWSE should have non-zero entries: the SBM has cycles, so k-step
    # self-return probabilities are non-trivial.
    assert (cache.full_rwse > 0).any(), "RWSE is all-zero; check graph or K."


def test_cache_precompute_is_idempotent(medium_graph):
    """Calling precompute() twice should not change cached values.

    This guards against a regression where someone makes precompute() do
    in-place updates or re-randomize on each call.
    """
    cache = FullGraphEncodingsCache(full_data=medium_graph, K=8)
    cache.precompute()
    first = cache.full_rwse.clone()
    cache.precompute()
    assert torch.equal(cache.full_rwse, first), (
        "Second precompute() changed cached RWSE. precompute() must be idempotent."
    )


def test_cache_slice_before_precompute_raises(medium_graph):
    """slice() before precompute() must fail loudly, not silently return None."""
    cache = FullGraphEncodingsCache(full_data=medium_graph, K=8)
    fake_ego = Data(x=medium_graph.x[:10], edge_index=torch.empty(2, 0, dtype=torch.long))
    fake_ego.n_id = torch.arange(10)
    with pytest.raises(RuntimeError, match="precompute"):
        cache.slice(fake_ego)


def test_cache_slice_without_n_id_raises(medium_graph):
    """slice() on a batch missing `n_id` must fail loudly.

    This catches the case where someone uses a non-NeighborLoader source
    (e.g. ClusterLoader, or raw subgraph) without attaching the global-id
    mapping the slice depends on.
    """
    cache = FullGraphEncodingsCache(full_data=medium_graph, K=8)
    cache.precompute()
    fake_ego = Data(x=medium_graph.x[:10], edge_index=torch.empty(2, 0, dtype=torch.long))
    # No n_id attribute on purpose.
    with pytest.raises(AttributeError, match="n_id"):
        cache.slice(fake_ego)


def test_cache_slice_returns_correct_rows(medium_graph):
    """slice(ego) must return cache.full_rwse[ego.n_id], not a recomputed RWSE.

    This is the cache's load-bearing correctness invariant. If someone changes
    slice() to recompute RWSE on the ego-net (which would be the "wrong" thing
    to do because RWSE on a truncated subgraph measures a different statistic
    than RWSE on the full graph), this test catches it.
    """
    cache = FullGraphEncodingsCache(full_data=medium_graph, K=8)
    cache.precompute()

    # Craft a fake ego-net with a known n_id.
    n_id = torch.tensor([3, 17, 92, 4, 55])
    fake_ego = Data(
        x=medium_graph.x[n_id],
        edge_index=torch.empty(2, 0, dtype=torch.long),
    )
    fake_ego.n_id = n_id

    rwse_ego, se_ego = cache.slice(fake_ego)
    assert se_ego is None  # no SE attached to cache
    assert torch.equal(rwse_ego, cache.full_rwse[n_id]), (
        "slice() did not return the expected rows of the cached RWSE."
    )


def test_cache_slice_with_se(medium_graph):
    """When the cache holds an SE tensor, slice() must return both PE and SE rows."""
    N = medium_graph.num_nodes
    fake_se = torch.randn(N, 4)
    cache = FullGraphEncodingsCache(full_data=medium_graph, K=8, se=fake_se)
    cache.precompute()

    n_id = torch.tensor([0, 50, 100])
    fake_ego = Data(
        x=medium_graph.x[n_id],
        edge_index=torch.empty(2, 0, dtype=torch.long),
    )
    fake_ego.n_id = n_id

    rwse_ego, se_ego = cache.slice(fake_ego)
    assert se_ego is not None
    assert torch.equal(rwse_ego, cache.full_rwse[n_id])
    assert torch.equal(se_ego, fake_se[n_id])


# ---------------------------------------------------------------------------
# Loader (SubgraphPreparer) tests
# ---------------------------------------------------------------------------
def test_loader_sample_batch_size(medium_graph):
    """sample_batch(batch_size=B) returns exactly B PreparedSubgraphs.

    Falls short only if the loader is exhausted, which on a 120-node SBM with
    batch_size=8 should not happen.
    """
    cache = FullGraphEncodingsCache(full_data=medium_graph, K=8)
    cache.precompute()
    loader = SubgraphPreparer(full_data=medium_graph, cache=cache)
    batches = loader.sample_batch(batch_size=8)
    assert len(batches) == 8
    assert all(isinstance(p, PreparedSubgraph) for p in batches)


def test_loader_prepared_subgraph_contract(medium_graph):
    """Each PreparedSubgraph from the loader satisfies the per-element shape contract.
    .source has matching .x and (sliced) .rwse rows
    .global_views and .local_views are non-empty
     each view's pe.size(0) == x.size(0)
    """
    cache = FullGraphEncodingsCache(full_data=medium_graph, K=8)
    cache.precompute()
    loader = SubgraphPreparer(full_data=medium_graph, cache=cache)
    batches = loader.sample_batch(batch_size=4)

    for prep in batches:
        assert prep.source.num_nodes == prep.rwse.size(0), (
            "PreparedSubgraph.rwse rows do not match source.num_nodes."
        )
        assert prep.rwse.size(1) == cache.K
        assert len(prep.global_views) > 0, "Empty global_views list."
        assert len(prep.local_views) > 0, "Empty local_views list."

        for v in (*prep.global_views, *prep.local_views):
            assert v.pe.size(0) == v.x.size(0), (
                "View pe and x have mismatched node counts."
            )


def test_loader_uses_cached_rwse(medium_graph):
    """The ego-net's RWSE on each PreparedSubgraph must equal the cache slice.

    This is the end-to-end version of `test_cache_slice_returns_correct_rows`:
    it verifies that the loader actually uses the cache (rather than e.g.
    silently falling back to per-ego RWSE recomputation).

    We rely on the fact that NeighborLoader attaches `n_id` to each sampled
    batch. We can't access that `n_id` directly from PreparedSubgraph (it's
    consumed and discarded inside the loader's iteration), so we verify the
    invariant indirectly: prep.rwse rows must exist somewhere in cache.full_rwse.
    """
    cache = FullGraphEncodingsCache(full_data=medium_graph, K=8)
    cache.precompute()
    loader = SubgraphPreparer(full_data=medium_graph, cache=cache)
    batches = loader.sample_batch(batch_size=4)

    full = cache.full_rwse  # (N, K)
    # For each prep, every row of prep.rwse must equal SOME row of full.
    # (Stronger than just "rows are a subset" -- we use exact equality.)
    for prep in batches:
        # (n_ego, 1, K) vs (1, N, K) -> (n_ego, N) bool of "row i equals row j of full".
        matches = (prep.rwse.unsqueeze(1) == full.unsqueeze(0)).all(dim=-1)
        per_row_matches = matches.any(dim=1)
        assert per_row_matches.all(), (
            f"{(~per_row_matches).sum().item()} ego rows do not appear in the "
            "cached full-graph RWSE. Loader is NOT using the cache."
        )


def test_loader_shuffle_actually_shuffles(medium_graph):
    """Two `sample_batch` calls should yield different seed nodes with high probability.

    NeighborLoader(shuffle=True) is the default; this test guards against a
    regression where someone hardcodes shuffle=False or fixes a seed.
    """
    cache = FullGraphEncodingsCache(full_data=medium_graph, K=8)
    cache.precompute()
    loader = SubgraphPreparer(full_data=medium_graph, cache=cache)

    batch_a = loader.sample_batch(batch_size=4)
    batch_b = loader.sample_batch(batch_size=4)

    # Compare the seed-region first node features as a cheap "did the sample
    # change" signal. We check that NOT ALL positions match -- a single
    # accidental collision would be fine, but identical batches mean no shuffle.
    all_same = all(
        torch.equal(a.source.x[0], b.source.x[0])
        for a, b in zip(batch_a, batch_b)
    )
    assert not all_same, (
        "Two sample_batch calls returned identical seed nodes at every "
        "position. NeighborLoader is not shuffling -- check shuffle=True."
    )
