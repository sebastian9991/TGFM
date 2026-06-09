import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import stochastic_blockmodel_graph

from tgfm.utils.diagnostics import (
    _lift_view_nodes_to_global,
    global_local_containment,
)
from tgfm.views.prepare import build_local_views_metis, compute_rwse

NUM_BLOCKS = 4
NODES_PER_BLOCK = 30  # total N = 120
P_INTRA = 0.25  # within-block edge probability
P_INTER = 0.02  # between-block edge probability
FEATURE_DIM = 16
SEED = 42




def _edges_as_frozenset(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    """Edges as a set of (min, max) tuples -- order-independent for undirected."""
    src, dst = edge_index.tolist()
    return {(min(s, d), max(s, d)) for s, d in zip(src, dst)}


@pytest.fixture(scope="session")
def medium_graph() -> Data:
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



def test_build_global_views_from_local(medium_graph):
    K = 5
    num_views = 1
    num_local_parts = 8
    rwse = compute_rwse(data=medium_graph, K=K)
    local_views = build_local_views_metis(
        data=medium_graph,
        rwse=rwse,
        num_parts=num_local_parts,
    )
    global_views = local_views[:num_views]
    local_view = local_views[num_views:]


    assert len(global_views) == num_views

    full_x = medium_graph.x
    full_edges = _edges_as_frozenset(medium_graph.edge_index)
    N = medium_graph.num_nodes

    # Lift each view back to global ids for downstream checks.
    view_global_ids: list[torch.Tensor] = []
    for view in global_views:
        assert view.x.size(0) == view.pe.size(
            0
        ), "view.x and view.pe must have matching node count."
        assert view.pe.size(1) == K, f"PE width must be K={K}, got {view.pe.size(1)}."
        assert view.edge_index.dim() == 2 and view.edge_index.size(0) == 2

        if view.edge_index.numel() > 0:
            assert view.edge_index.max().item() < view.x.size(
                0
            ), "Relabeled edge id exceeds local node count -- relabeling bug."
            assert view.edge_index.min().item() >= 0

        view_global_ids.append(_lift_view_nodes_to_global(view, full_x))

    # ---- 1. Nodes exist in the larger graph ----------------------------
    for global_ids in view_global_ids:
        assert (global_ids >= 0).all() and (global_ids < N).all()
        assert (
            global_ids.unique().numel() == global_ids.numel()
        ), "A single view contains the same original node twice."

    # ---- 2. Edges exist in the larger graph ----------------------------
    # Lift local edge ids back to global via the per-view id mapping.
    for view, global_ids in zip(global_views, view_global_ids):
        if view.edge_index.numel() == 0:
            continue
        global_src = global_ids[view.edge_index[0]]
        global_dst = global_ids[view.edge_index[1]]
        view_global_edges = {
            (min(int(s), int(d)), max(int(s), int(d)))
            for s, d in zip(global_src.tolist(), global_dst.tolist())
        }
        missing = view_global_edges - full_edges
        assert not missing, (
            f"View has {len(missing)} edges not present in the original graph "
            f"(first few: {list(missing)[:5]})."
        )
    # ---- 3. Views form a COVER, not a partition ------------------------
    #   (a) Not every node appears in the global partition
    #   (b) some nodes appear in multiple views (because of 1-hop expansion)
    # NOT "every node appears in exactly one view."
    appearance_count = torch.zeros(N, dtype=torch.long)
    for global_ids in view_global_ids:
        appearance_count[global_ids] += 1

    assert (appearance_count == 0).any(), (
        "It is unlikely for each node to exist in the global partition. This is discouraged."
    )

    # ---- 4. Overlap ratio is bounded -----------------------------------
    # What is a correct overlap ratio?
    total_mass = int(appearance_count.sum().item())
    overlap_ratio = total_mass / N
    assert overlap_ratio < num_views, (
        f"Sum of view sizes is {total_mass} = {overlap_ratio:.2f} * N. "
        f"Expected < 3.0 -- if higher, the 1-hop expansion is pulling in many nodes. "
    )
    assert overlap_ratio < 1.0
    print(f'OVERLAP RATIO: {overlap_ratio}')


def test_global_local_overlap_from_local(medium_graph):
    K = 5
    num_views = 1
    num_local_parts = 8
    rwse = compute_rwse(data=medium_graph, K=K)
    local_views = build_local_views_metis(
        data=medium_graph,
        rwse=rwse,
        num_parts=num_local_parts,
    )
    global_views = local_views[:num_views]
    local_view = local_views[num_views:]

    stats = global_local_containment(global_views, local_views, medium_graph.x)

    # ---- Existence / range ---------------------------------------------
    assert 0.0 <= stats["mean_containment"] <= 1.0
    assert 0.0 <= stats["max_containment"] <= 1.0
    assert 0.0 <= stats["mean_jaccard"] <= 1.0
    assert stats["pair_table"].shape == (len(global_views), len(local_views))

    # ---- Non-trivial overlap -------------------------------------------
    # Globals cover ~70% of the graph; locals cover ~25% + boundary. So on
    # average at least some local nodes should land inside some global.
    assert stats["mean_containment"] > 0.1, (
        f"Globals and locals barely overlap (mean_containment="
        f"{stats['mean_containment']:.3f}). View construction may be broken."
    )

    # ---- Locals not fully nested ---------------------------------------
    # If this fires it's the degeneracy signal: every local view is a subset
    # of every global view, so the prediction loss has no novel local info.
    assert stats["mean_containment"] < 0.99, (
        f"Locals are nearly fully contained in globals (mean_containment="
        f"{stats['mean_containment']:.3f}). Risks degenerate prediction task."
    )

    # ---- max ≥ mean ----------------------------------------------------
    # Definitional: the best-matching global covers at least as much of each
    # local as the average global.
    assert stats["max_containment"] >= stats["mean_containment"] - 1e-6

    print(f'MEAN_CONTAINMENT: {stats["mean_containment"]}\n')
    print(f'MAX_CONTAINMENT: {stats["max_containment"]}\n')
    print(f'MEAN_JACCARD: {stats["mean_jaccard"]}\n')
