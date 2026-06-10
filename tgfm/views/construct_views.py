"""Construct views helper functions."""

from typing import List, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter

from tgfm.models.mpnn import GCNEncoder
from tgfm.views.augmentations import augment
from tgfm.views.prepare import PreparedSubgraph


def embed_views_batched(
    encoder: torch.nn.Module,
    views: List[Data],
    edge_drop_rate: float,
    feat_mask_rate: float,
    augment_views: bool = False,
) -> Tensor:
    """Encode a list of view-Data objects and mean-pool each to a single vector.

    Args:
        encoder: GraphGPSEncoder (shared across global & local views).
        views:   list of length N_views, each a PyG Data with x, edge_index,
                 and optionally `pe` and/or `se` attributes carrying the
                 precomputed positional/structural encodings.
        edge_drop_rate: Drop rate for adjacency matrix augmentation.
        feat_mask_rate: Masking rate for feature augmentation.
        augment_views: Whether to augment the views.

    Returns:
        z: (N_views, d)
    """
    if len(views) == 0:
        raise ValueError('embed_views_batched got an empty list of views.')

    if isinstance(encoder, GCNEncoder):
        if augment_views:
            aug = []
            for v in views:
                x_v, ei_v = augment(v.x, v.edge_index, edge_drop_rate, feat_mask_rate)
                augmented_view = Data(x=x_v, edge_index=ei_v)
                if getattr(v, 'pe', None) is not None:
                    augmented_view.pe = v.pe
                if getattr(v, 'se', None) is not None:
                    augmented_view.se = v.se
                aug.append(augmented_view)
                views = aug
            batch = Batch.from_data_list(views)
            h = encoder(
                x=batch.x,
                edge_index=batch.edge_index,
            )
        else:
            batch = Batch.from_data_list(views)
            h = encoder(
                x=batch.x,
                edge_index=batch.edge_index,
            )
    else:
        batch = Batch.from_data_list(views)
        pe = getattr(batch, 'pe', None)
        se = getattr(batch, 'se', None)
        edge_attr = getattr(batch, 'edge_attr', None)
        h = encoder(
            x=batch.x,
            pe=pe,
            se=se,
            edge_index=batch.edge_index,
            batch=batch.batch,
            edge_attr=edge_attr,
        )  # (sum_N, d)

    # Mean-pool per view (= per graph in this Batch).
    # TODO: Check this mean-pooling methods correctness.
    z = scatter(h, batch.batch, dim=0, reduce='mean')  # (N_views, d)
    return z


def _augment_view(v: Data, edge_drop_rate: float, feat_mask_rate: float) -> Data:
    x_v, ei_v = augment(v.x, v.edge_index, edge_drop_rate, feat_mask_rate)
    nv = Data(x=x_v, edge_index=ei_v)
    nv.n_id = v.n_id  # node set/order unchanged by augment
    if getattr(v, 'pe', None) is not None:
        nv.pe = v.pe
    if getattr(v, 'se', None) is not None:
        nv.se = v.se
    return nv


def embed_views_nodelevel(
    encoder: torch.nn.Module,
    views: List[Data],
    edge_drop_rate: float,
    feat_mask_rate: float,
    augment_views: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Encode views WITHOUT pooling. Each view augmented independently.

    Returns:
        h:       (M, d) per-node embeddings, M = total nodes across all views
        node_id: (M,)   original node id per row
        view_id: (M,)   which view each row came from
    """
    if augment_views and isinstance(encoder, GCNEncoder):
        views = [_augment_view(v, edge_drop_rate, feat_mask_rate) for v in views]

    batch = Batch.from_data_list(views)
    h = encoder(x=batch.x, edge_index=batch.edge_index)  # (M, d) — no scatter/pool

    device = h.device
    node_id = torch.cat([v.n_id.to(device) for v in views])
    view_id = torch.cat(
        [
            torch.full((v.num_nodes,), i, dtype=torch.long, device=device)
            for i, v in enumerate(views)
        ]
    )
    assert h.size(0) == node_id.size(0) == view_id.size(0)
    return h, node_id, view_id


def build_views(
    encoder: torch.nn.Module,
    global_views: List[Data],  # length B * V_g (flattened across batch)
    local_views: List[Data],  # length B * V_l
    B: int,
    V_g: int,
    V_l: int,
    edge_drop_rate: float,
    feat_mask_rate: float,
    augment_views: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Encode all global+local views and reshape to (B, V_g, d), (B, V_l, d).

    Convention: global_views is ordered [subgraph 0's V_g views, subgraph 1's V_g views, ...].
                Same for local_views with V_l.
    """
    assert len(global_views) == B * V_g, (len(global_views), B, V_g)
    assert len(local_views) == B * V_l, (len(local_views), B, V_l)

    # Encode in one combined forward to keep BN/attention stats consistent.
    all_views = global_views + local_views
    z_all = embed_views_batched(
        encoder, all_views, edge_drop_rate, feat_mask_rate, augment_views
    )  # (B*(V_g+V_l), d)
    d = z_all.size(-1)

    z_global = z_all[: B * V_g].view(B, V_g, d)
    z_local = z_all[B * V_g :].view(B, V_l, d)
    return z_global, z_local


def flatten_views(
    batch: List[PreparedSubgraph],
) -> tuple[list[Data], list[Data], int, int, int]:
    """Flatten the per-subgraph view lists into two lists ordered by subgraph.

    Returns:
        global_views: list of B*V_g Data
        local_views:  list of B*V_l Data   (note: V_l may differ per subgraph;
                                            we truncate to the min to keep tensors uniform).
        B, V_g, V_l
    """
    B = len(batch)
    V_g = min(len(p.global_views) for p in batch)
    V_l = min(len(p.local_views) for p in batch)
    if V_g == 0 or V_l == 0:
        raise RuntimeError(
            f'Got V_g={V_g}, V_l={V_l}; one or more subgraphs produced zero views. '
            'Check num_local_parts / coverage_frac vs. subgraph sizes.'
        )

    global_views: list[Data] = []
    local_views: list[Data] = []
    for p in batch:
        global_views.extend(p.global_views[:V_g])
        local_views.extend(p.local_views[:V_l])
    return global_views, local_views, B, V_g, V_l
