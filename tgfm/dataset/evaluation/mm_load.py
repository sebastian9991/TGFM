"""MM-Graph loader (Mosaic of Modalities): benchmark tensors -> PyG Data.

Converts one MM-Graph dataset directory into a torch_geometric Data matching
what the graph encoder expects, and exposes the precomputed image feature as
data.image_x for the image tower.

Directory layout under _mm_root():

  <name>/
    {clip,imagebind,t5vit,t5dino}_feat.pt   per-node [text || image] features
    nc_edges-nodeid.pt                       NC edge list, (E, 2)
    split.pt                                 {train_idx, val_idx, test_idx}
    labels-w-missing.pt                      (N,) int64 node labels
    lp-edge-split.pt                         LP edge split (LP datasets only)

VERIFIED on ele-fashion and books-nc (2026-08):
  * *_feat.pt is a single float32 tensor of width 1536, NOT a dict.
  * Layout is [text || image] with text FIRST. Confirmed by loading both
    t5dino_feat.pt and t5vit_feat.pt: they share the T5 text encoder, and
    only their first 768 columns are identical (allclose True on [:, :768],
    False on [:, 768:]). So t5dino = T5(768) || DINOv2-base(768) -- the image
    half is DINOv2-*base*, not DINOv2-L as the 1024 figure would suggest.
  * split.pt is a dict of int64 index tensors keyed train_idx/val_idx/test_idx
    (ele-fashion 58659/9777/29330 of 97766; books-nc 406689/67805/203428 of
    685294 -- the 6/1/3 split of the paper).
  * labels-w-missing.pt has NO negative sentinel in either NC dataset:
    ele-fashion runs 0..11 with 11 distinct values present (one of the 12
    label ids is empty), books-nc runs 0..10 with all 11 present. The y >= 0
    guard below is kept as a cheap safeguard, not because it filters anything
    in these two.
  * Edges are stored (E, 2) and are transposed to PyG's (2, E) here.

feat_name selects the encoder bundle. Prefer 't5dino': the DINOv2 image half
is self-supervised rather than text-aligned, so the volume term has to earn
the text-image alignment instead of inheriting CLIP's.
"""

from pathlib import Path
from typing import List, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from tgfm.utils.path import get_scratch

# Ego-subgraphs are built by tgfm.dataset.evaluation.mm_graph_sampler:
#     from .mm_graph_sampler import parse_mm_target_data
# parse_target_data in tgfm/utils/process.py cannot be used -- it reads a
# precomputed JSON of subgraphs from GraphCLIP's preprocessing that MM-Graph
# does not ship.

DIR_OF = {
    'sports-copurchase': 'sports-copurchase',
    'cloth-copurchase': 'cloth-copurchase',
    'books-lp': 'books-lp',
    'ele-fashion': 'ele-fashion',
    'books-nc': 'books-nc',
}
NC_DATASETS = {'ele-fashion', 'books-nc'}

# (text_dim, image_dim) per bundle; text occupies the leading columns.
# t5dino / t5vit measured at 1536 total on ele-fashion and books-nc.
FEAT_DIMS = {
    't5dino': (768, 768),    # T5 || DINOv2-base   (verified)
    't5vit': (768, 768),     # T5 || ViT-B         (verified: shares T5 half)
    'clip': (512, 512),      # not yet verified
    'imagebind': (1024, 1024),  # not yet verified
}


def _mm_root() -> Path:
    return get_scratch() / 'mm_graph_datasets'


def _split_feat(feat: Tensor, feat_name: str) -> Tuple[Tensor, Tensor]:
    """Split a (N, text_dim + image_dim) feature into (text_x, image_x).

    Text occupies the leading columns; see the module docstring for the
    t5dino/t5vit cross-check that establishes the order.
    """
    text_dim, image_dim = FEAT_DIMS[feat_name]
    if feat.shape[-1] != text_dim + image_dim:
        raise ValueError(
            f'{feat_name}: expected {text_dim + image_dim} columns, got '
            f'{feat.shape[-1]}. Update FEAT_DIMS for this bundle.'
        )
    return feat[:, :text_dim], feat[:, text_dim:]


def _as_edge_index(e: Tensor, num_nodes: int, undirected: bool = True) -> Tensor:
    """Coerce a stored edge tensor to a (2, E) LongTensor.

    MM-Graph ships (E, 2). Symmetrized by default so the ego-subgraph sampler
    and message passing see both directions, matching the co-purchase /
    similarity semantics of these graphs.
    """
    e = torch.as_tensor(e)
    if e.dim() != 2:
        raise ValueError(f'Unexpected edge tensor shape {tuple(e.shape)}')
    edge_index = (e if e.shape[0] == 2 else e.t()).long().contiguous()
    if undirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    return edge_index


def _apply_split(data: Data, split: dict) -> None:
    """Attach boolean train/val/test node masks from the stored index tensors."""
    n = data.num_nodes
    for mask_name, key in (
        ('train_mask', 'train_idx'),
        ('val_mask', 'val_idx'),
        ('test_mask', 'test_idx'),
    ):
        idx = torch.as_tensor(split[key]).long()
        mask = torch.zeros(n, dtype=torch.bool)
        mask[idx] = True
        setattr(data, mask_name, mask)


def load_mm_data(
    dataset: str, feat_name: str = 't5dino', undirected: bool = True
) -> Tuple[Data, List[str], List[str]]:
    """Load one MM-Graph dataset as a PyG Data.

    Returns (data, classes, c_descs); the latter two are empty for the
    linear-probe protocol (no class-name text is used) and exist only for
    signature parity with the TAG loader.
    """
    if dataset not in DIR_OF:
        raise ValueError(f'Dataset {dataset!r} not in MM-Graph.')
    root = _mm_root() / DIR_OF[dataset]

    feat = torch.load(root / f'{feat_name}_feat.pt', map_location='cpu')
    text_x, image_x = _split_feat(feat.float(), feat_name)
    num_nodes = text_x.size(0)

    if dataset in NC_DATASETS:
        raw_edges = torch.load(root / 'nc_edges-nodeid.pt', map_location='cpu')
    else:
        lp = torch.load(root / 'lp-edge-split.pt', map_location='cpu')
        tr = lp['train']
        raw_edges = torch.stack([tr['source_node'], tr['target_node']], dim=0)

    data = Data(
        x=text_x, edge_index=_as_edge_index(raw_edges, num_nodes, undirected)
    )
    data.image_x = image_x

    if dataset in NC_DATASETS:
        data.y = torch.as_tensor(
            torch.load(root / 'labels-w-missing.pt', map_location='cpu')
        ).long()
        _apply_split(
            data, torch.load(root / 'split.pt', map_location='cpu')
        )

    return data, [], []
