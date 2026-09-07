"""Emit frozen LeGTJEPA node features for a target graph.

Runs encode_node once per node -- each node via its own ego-subgraph, so every
node is the center exactly once and gets a full-context embedding -- and
collects the center rows into an N x d matrix. Optionally concatenates the
projected text and image embeddings (the initialization ablation):

    node               : projected graph-tower center-node state         (d)
    node_text          : [node || projected text]                       (2d)
    node_text_image    : [node || projected text || projected image]    (3d)

The matrix is cached to disk; the LP training loop reads it as data.x. Emission
is a single pass, so the per-node ego-subgraph cost is paid once per graph, not
once per training epoch.
"""

import logging
from pathlib import Path

import torch
from torch import Tensor
from torch_geometric.loader import DataLoader

from tgfm.dataset.evaluation.mm_load import load_mm_data
from tgfm.models.legtjepa import LeGTJEPA
from tgfm.utils.args import LeGTJEPAArguments
from tgfm.utils.mm_sampler import parse_mm_target_data


@torch.no_grad()
def emit_node_features(
    model: LeGTJEPA,
    dataset: str,
    feat_name: str,
    emit_mode: str,
    device: torch.device,
    batch_size: int,
    num_workers: int = 8,
) -> Tensor:
    """N x d' frozen feature matrix, row u = node u's representation."""
    was_training = model.training
    model.eval()
    try:
        data, _, _ = load_mm_data(dataset, feat_name=feat_name)
        graphs = parse_mm_target_data(dataset, data)
        graphs.assert_alignment()
        loader = DataLoader(
            graphs,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
        rows = []
        for batch in loader:
            batch = batch.to(device)
            z_node = model.encode_node(batch)  # (B, d), projected center state
            if emit_mode in ('node_text', 'node_text_image'):
                text_x = batch.x[batch.root_n_index]
                z_text = model.encode_text_features(text_x)
                z_node = torch.cat([z_node, z_text], dim=-1)
            if emit_mode == 'node_text_image':
                z_image = model.encode_image(batch.image_x)
                z_node = torch.cat([z_node, z_image], dim=-1)
            rows.append(z_node.cpu())
        feats = torch.cat(rows, dim=0)
        logging.info(
            'Emitted %s features for %s: %s (mode=%s)',
            emit_mode,
            dataset,
            tuple(feats.shape),
            emit_mode,
        )
        return feats
    finally:
        if was_training:
            model.train()


def emit_or_load(
    model: LeGTJEPA,
    dataset: str,
    model_args: LeGTJEPAArguments,
    emit_mode: str,
    device: torch.device,
    cache_dir: Path,
    batch_size: int,
) -> Tensor:
    """Return the cached feature matrix, emitting and caching it if absent."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f'{dataset}--{model_args.mm_feat_name}--{emit_mode}.pt'
    if cache.exists():
        logging.info('Loading cached features %s', cache)
        return torch.load(cache, map_location='cpu')
    feats = emit_node_features(
        model, dataset, model_args.mm_feat_name, emit_mode, device, batch_size
    )
    torch.save(feats, cache)
    logging.info('Cached features to %s', cache)
    return feats
