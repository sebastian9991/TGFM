"""Graph-tower node features for the MM-Graph modality ablation.

The MM-Graph loader returns a full-graph ``Data`` with ``x`` = T5 text feature
and ``image_x`` = DINOv2 image feature, and the ego-subgraph sampler hands the
graph tower ``batch.x``. ``graph_feat`` decides what that is:

    'text'        x = T5                   G-T, G-T-I
    'image'       x = DINOv2               G-I (no text anywhere in the model)
    'text_image'  x = [T5 || DINOv2]       G-T-I with both as node input

The swap happens on the full graph *before* the sampler is built, so every node
of every subgraph carries the selected feature, and it is applied identically
in pretraining (mm_main.load_source_graphs) and in the probe
(mm_linear_probe.evaluate_dataset). Partner targets are unaffected: the image
target is the sampler's center ``image_x``; the text target is the first
``text_in_dim`` columns of the center row of ``batch.x``.
"""

from typing import Any

import torch
from torch import Tensor
from torch_geometric.data import Data

from tgfm.utils.args import LeGTJEPAArguments


def select_graph_features(data: Data, model_args: LeGTJEPAArguments) -> Data:
    """Set ``data.x`` to the graph tower's node features, validating widths."""
    text_x, image_x = data.x, data.image_x
    if text_x.size(1) != model_args.text_in_dim:
        raise ValueError(
            f'text feature is {text_x.size(1)}-d, text_in_dim={model_args.text_in_dim}'
        )
    if image_x.size(1) != model_args.image_in_dim:
        raise ValueError(
            f'image feature is {image_x.size(1)}-d, '
            f'image_in_dim={model_args.image_in_dim}'
        )
    if model_args.graph_feat == 'image':
        data.x = image_x
    elif model_args.graph_feat == 'text_image':
        data.x = torch.cat((text_x, image_x), dim=1)
    if data.x.size(1) != model_args.graph_in_dim:
        raise ValueError(
            f'graph node features are {data.x.size(1)}-d, '
            f'graph_in_dim={model_args.graph_in_dim}'
        )
    return data


def check_graph_features(ds: Any, data: Data) -> None:
    """Fail if the sampler does not serve the (possibly swapped) ``data.x``.

    Guards against a sampler that caches subgraph features by dataset name or
    copies ``x`` before the swap: item 0 is node 0's ego-subgraph, so its
    center row must equal ``data.x[0]``.
    """
    sub = ds[0]
    root = int(torch.as_tensor(sub.root_n_index).view(-1)[0])
    if not torch.equal(sub.x[root].cpu(), data.x[0].cpu()):
        raise RuntimeError(
            'Ego-subgraph center features do not match data.x after '
            'select_graph_features; the sampler is not reading data.x at item time.'
        )


def center_text_feature(batch: Any, model_args: LeGTJEPAArguments) -> Tensor:
    """Text target: center-node T5 row, (B, text_in_dim). Copies (advanced index)."""
    return batch.x[batch.root_n_index, : model_args.text_in_dim]
