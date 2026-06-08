"""Training loop for LeGraph.

Key design points:
    - Single shared encoder (no EMA, no teacher-student, no predictor).
    - Per-step: prep views for B subgraphs, encode all views in one batched
      forward, compute predictive and SIGReg loss, backprop, step.
    - Logging: every `log_freq` steps, run encoder over a held-out probe batch
      and logging collapse diagnostics.

View construction & encoding utilities.

These mirror the pseudo-code:
    - embed_subgraph_region(...)  -> z (d,)
    - build_views(...)            -> (z_global, z_local)

The encoder is *shared* across all views (no teacher-student).

Each view in the input batch is represented as a PyG-style `Data` (or a dict)
with the fields:
    x:          (|V|, in_dim)         node features for this view
    edge_index: (2, |E|)               edges within the view
    pe:         (|V|, K)               positional encoding (e.g. RWSE) restricted to view nodes
    edge_attr:  (|E|, ?)               optional, edge features

To run efficiently we batch all views of all subgraphs in one `Batch`. We then
mean-pool by `batch` to get one (d,) embedding per view, and reshape back to
(B, V_g, d) and (B, V_l, d) via the bookkeeping returned from the dataloader.
"""

import argparse
import logging
import random
from pathlib import Path
from typing import Optional

import torch
import torch_geometric.transforms as T
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.data import Data
from tqdm import tqdm

from tgfm.dataset.evaluation.linear_prob_pyg import (
    compute_node_embeddings,
    evaluate_linear_probe,
    node_classification,
)
from tgfm.dataset.pyg.data import (
    FullGraphEncodingsCache,
    SubgraphPreparer,
    get_dataset,
)
from tgfm.models.graphgps import GraphGPSEncoder, LinearEncoder
from tgfm.models.leJepa_loss import LeJEPALoss
from tgfm.utils.args import (
    DataArguments,
    GraphGPSArguments,
    MetaArguments,
    ModelArguments,
    parse_args,
)
from tgfm.utils.diagnostics import collapse_diagnostics
from tgfm.utils.logger import setup_logging
from tgfm.utils.path import get_root_dir
from tgfm.utils.seed import seed_everything
from tgfm.views.construct_views import build_views, flatten_views

parser = argparse.ArgumentParser(
    description='Distributed step-based pretraining UniGraph.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file', type=str, required=True, help='Path to configuration file.'
)


def save_checkpoint(
    path: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    total_loss: float,
    prediction_loss: float,
    sig_loss: float,
) -> None:
    """Save a checkpoint. Called only on rank 0.

    Uses tmp-file + rename for atomicity: if the job dies mid-save, the old
    checkpoint is still valid and the partial .tmp file is just debris.
    """
    rng_states = {
        'python': random.getstate(),
        'torch_cpu': torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_states['torch_cuda_all'] = torch.cuda.random.get_rng_state_all()

    payload = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_loss': total_loss,
        'prediction_loss': prediction_loss,
        'sig_loss': sig_loss,
    }
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp_path)
    tmp_path.rename(path)


def evaluate(
    embeddings: Tensor,
    full_data: Data,
    repeat: int,
    data_random_seed: int,
    dataset: str,
    full_eval: bool = False,
) -> dict:
    results = evaluate_linear_probe(
        embeddings,
        full_data,
        repeat=repeat,
        data_random_seed=data_random_seed,
    )
    has_masks = (
        hasattr(full_data, 'train_mask')
        and hasattr(full_data, 'val_mask')
        and hasattr(full_data, 'test_mask')
        and full_data.train_mask is not None
    )
    if has_masks:
        masks = (full_data.train_mask, full_data.val_mask, full_data.test_mask)
    else:
        masks = None

    if full_eval:
        node_classification(Z=embeddings, Y=full_data.y, dataset=dataset, masks=masks)

    return results


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    meta_args: MetaArguments,
    save_dir: Path,
) -> None:
    assert isinstance(model_args, GraphGPSArguments)

    # 1. Load dataset
    transform = T.Compose(
        [
            T.NormalizeFeatures(),
            T.ToUndirected(),
            T.AddSelfLoops(),
            T.RemoveIsolatedNodes(),
        ]
    )
    dataset = get_dataset(
        root=str(meta_args.root_dir), name=data_args.data_name, transform=transform
    )
    full_data = dataset[0]
    assert full_data.has_isolated_nodes() == False
    assert full_data.has_self_loops() == True
    assert full_data.is_directed() == False
    logging.info(
        'Loaded %s: N=%d nodes, E=%d edges, in_dim=%d',
        data_args.data_name,
        full_data.num_nodes,
        full_data.num_edges,
        full_data.num_features,
    )

    # 2. Precompute RWSE on the full graph
    # TODO: We may need to cache this result if it takes too long
    cache = FullGraphEncodingsCache(full_data, K=model_args.rwse_K)
    logging.info('Precomputing RWSE (K=%d) on the full graph...', model_args.rwse_K)
    cache.precompute()
    logging.info('Computed RWSE.')

    # 3. Build the stream preparer
    preparer = SubgraphPreparer(
        full_data=full_data,
        cache=cache,
        num_neighbors=model_args.num_neighbors,
        seed_batch_size=1,
        prepare_kwargs=dict(
            num_local_parts=model_args.num_local_parts,
            num_global_views=model_args.num_global_views,
            global_coverage_frac=model_args.global_coverage_frac,
            global_strategy=model_args.global_strategy,
        ),
    )
    logging.info('Streaming preparer loaded.')

    # 4. Build dataset-specific encoders
    node_encoder = LinearEncoder(
        in_dim=full_data.num_features, out_dim=model_args.node_out_dim
    )
    # TODO:  Should it be an MLP?
    pe_encoder = LinearEncoder(in_dim=model_args.rwse_K, out_dim=model_args.pe_out_dim)
    se_encoder: Optional[torch.nn.Module] = None

    encoder = GraphGPSEncoder(
        node_encoder=node_encoder,
        pe_encoder=pe_encoder,
        se_encoder=se_encoder,
        dim=model_args.dim,
        num_layers=model_args.num_layers,
        num_heads=model_args.num_heads,
        dropout=model_args.dropout,
        local_gnn_type=model_args.local_gnn_type,
        attn_type=model_args.attn_type,
        norm=model_args.norm,
    ).to(model_args.device)

    loss_fn = LeJEPALoss(
        lambd=model_args.lambd,
        num_slices=model_args.num_slices,
    ).to(model_args.device)

    optim = AdamW(
        encoder.parameters(), lr=model_args.lr, weight_decay=model_args.weight_decay
    )

    # Recommended from LeJEPA paper.
    scheduler = CosineAnnealingWarmRestarts(
        optimizer=optim, T_0=20
    )  # TODO: Paramaterize this.

    logging.info(
        'Encoder params: %.2fM',
        sum(p.numel() for p in encoder.parameters()) / 1e6,
    )

    eval_freq = model_args.eval_frequency
    eval_repeat = model_args.eval_repeat
    best_loss: float = float('inf')
    for step in tqdm(range(model_args.num_steps), desc='Training steps'):
        batch = preparer.sample_batch(model_args.batch_size)
        if len(batch) < model_args.batch_size:
            logging.warning(
                'Loader exhausted before filling batch; got %d / %d',
                len(batch),
                model_args.batch_size,
            )

        if len(batch) == 0:
            break

        global_views, local_views, B, V_g, V_l = flatten_views(batch)

        # Move all views to the encoder's device.
        global_views = [g.to(model_args.device) for g in global_views]
        local_views = [l.to(model_args.device) for l in local_views]

        z_global, z_local = build_views(
            encoder=encoder,
            global_views=global_views,
            local_views=local_views,
            B=B,
            V_g=V_g,
            V_l=V_l,
        )

        out = loss_fn(z_global, z_local)

        optim.zero_grad(
            set_to_none=True
        )  # Is set to none correct here? Should it be set to zero by default?
        out.total.backward()
        optim.step()
        scheduler.step()

        if out.total < best_loss:
            save_checkpoint(
                save_dir, step, encoder, optim, out.total, out.pred, out.sigreg
            )
            logging.info(
                f'[step {step + 1}] New best loss {out.total:4f} saved best.pt'
            )

        assert cache.full_rwse is not None
        if step % model_args.log_frequency == 0 or step == model_args.num_steps - 1:
            with torch.no_grad():
                # Diagnostics on the flattened global-view embeddings.
                stats = collapse_diagnostics(z_global.reshape(-1, z_global.size(-1)))
            logging.info(
                'step=%d loss=%.4f pred=%.4f sigreg=%.4f | '
                'rank=%.0f eff_rank=%.1f trace=%.3f var[min/mean/max]=%.3f/%.3f/%.3f '
                'cos[mean/std]=%.3f/%.3f',
                step,
                out.total.item(),
                out.pred.item(),
                out.sigreg.item(),
                stats.rank,
                stats.eff_rank,
                stats.trace_cov,
                stats.var_min,
                stats.var_mean,
                stats.var_max,
                stats.cos_mean,
                stats.cos_std,
            )

        if model_args.eval_frequency > 0 and step > 0 and step % eval_freq == 0:
            logging.info(
                'step=%d running linear prob (repeat = %d)...', step, eval_repeat
            )
            embeddings = compute_node_embeddings(
                encoder,
                full_data,
                rwse=cache.full_rwse,
                se=cache.full_se,
                device=model_args.device,
            )
            results = evaluate(
                embeddings,
                full_data,
                repeat=eval_repeat,
                data_random_seed=meta_args.global_seed,
                dataset=data_args.data_name,
            )
            logging.info(
                'step=%d prob[%s] acc=%.4f +/- %4.f over %d splits',
                step,
                results['probe_type'],
                results['mean'],
                results['std'],
                len(results['accuracies']),
            )

        logging.info('Final evaluation (repeat=%d)', eval_repeat)
        embeddings = compute_node_embeddings(
            encoder,
            full_data,
            rwse=cache.full_rwse,
            se=cache.full_se,
            device=model_args.device,
        )
        results = evaluate(
            embeddings,
            full_data,
            repeat=eval_repeat,
            data_random_seed=meta_args.global_seed,
            dataset=data_args.data_name,
            full_eval=True,
        )
        logging.info(
            'FINAL step=%d prob[%s] acc=%.4f +/- %4.f over %d splits',
            step,
            results['probe_type'],
            results['mean'],
            results['std'],
            len(results['accuracies']),
        )


def main() -> None:
    root = get_root_dir()
    args = parser.parse_args()
    config_file_path = root / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)
    root_dir = Path(str(meta_args.root_dir))
    seed_everything(meta_args.global_seed)
    setup_logging(meta_args.log_file_path)
    for experiment, experiment_arg in experiment_args.exp_args.items():
        train(
            model_args=experiment_arg.model_args,
            data_args=experiment_arg.data_args,
            meta_args=meta_args,
            save_dir=root_dir / 'weights',
        )


if __name__ == '__main__':
    main()
