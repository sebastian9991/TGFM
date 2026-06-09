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
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch_geometric.data import Data
from tqdm import tqdm

from tgfm.dataset.evaluation.linear_prob_pyg import (
    evaluate_linear_probe,
    node_classification,
)
from tgfm.dataset.pyg.data import get_dataset
from tgfm.models.leJepa_loss import LeJEPALoss
from tgfm.models.mpnn import build_encoder
from tgfm.utils.args import (
    DataArguments,
    MetaArguments,
    ModelArguments,
    SimpleMPNN,
    parse_args,
)
from tgfm.utils.diagnostics import collapse_diagnostics
from tgfm.utils.logger import setup_logging
from tgfm.utils.path import get_root_dir
from tgfm.utils.seed import seed_everything
from tgfm.views.construct_views import build_views, flatten_views
from tgfm.views.prepare import prepare_subgraph

parser = argparse.ArgumentParser(
    description='Distributed step-based pretraining UniGraph.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file', type=str, required=True, help='Path to configuration file.'
)
parser.add_argument('--eval', type=str, default='yours', choices=['yours', 'paper'])


@torch.no_grad()
def encode_full_graph(
    encoder: torch.nn.Module, x: Tensor, edge_index: Tensor
) -> Tensor:
    """Embed every node of the full (un-augmented) graph in eval mode."""
    encoder.eval()
    return encoder(x, edge_index)


def get_masks(full_data: Data, dataset: str) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
    """Return (train, val, test) masks if present, else None.

    WikiCS train/val masks are 2-D (N, 20); test is 1-D. Cora/CiteSeer/PubMed
    are 1-D. Computer/CoauthorCS have no public masks -> None (random split).
    """
    if dataset == 'WikiCS':
        return (full_data.train_mask, full_data.val_mask, full_data.test_mask)
    has = (
        getattr(full_data, 'train_mask', None) is not None
        and getattr(full_data, 'val_mask', None) is not None
        and getattr(full_data, 'test_mask', None) is not None
    )
    if not has:
        return None
    return (full_data.train_mask, full_data.val_mask, full_data.test_mask)


def run_eval(
    embeddings: Tensor,
    full_data: Data,
    model_args: ModelArguments,
    data_args: DataArguments,
    meta_args: MetaArguments,
    which: str,
) -> None:
    assert isinstance(model_args, SimpleMPNN)
    if which == 'paper':
        from tgfm.dataset.evaluation import ssge_eval

        masks = get_masks(full_data, data_args.data_name)
        ssge_eval.node_classification(
            embeddings,
            full_data.y,
            dataset=data_args.data_name,
            masks=masks,
            n_repeats=model_args.eval_repeat,
            lr=model_args.lr2,
            wd=model_args.wd2,
        )
        ssge_eval.node_clustering(
            embeddings, full_data.y, n_repeats=model_args.eval_repeat
        )
    else:
        # Your pipeline. Matches the call in your main.py.
        from tgfm.dataset.evaluation.linear_prob_pyg import (
            evaluate_linear_probe,
        )

        results = evaluate_linear_probe(
            embeddings,
            full_data,
            repeat=model_args.eval_repeat,
            data_random_seed=meta_args.global_seed,
        )
        logging.info(
            'probe[%s] acc=%.4f +/- %.4f over %d splits',
            results['probe_type'],
            results['mean'],
            results['std'],
            len(results['accuracies']),
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

    if full_eval:
        logging.info(f'Full Evaluation.')
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
        node_classification(Z=embeddings, Y=full_data.y, dataset=dataset, masks=masks)

    return results


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    meta_args: MetaArguments,
    save_dir: Path,
    eval_mode: str = 'yours',
) -> None:
    assert isinstance(model_args, SimpleMPNN)
    device = model_args.device

    dataset = get_dataset(root=str(meta_args.root_dir), name=data_args.data_name)
    full_data = dataset[0]
    logging.info(
        'Loaded %s: N=%d nodes, E=%d edges, in_dim=%d',
        data_args.data_name,
        full_data.num_nodes,
        full_data.num_edges,
        full_data.num_features,
    )

    x = full_data.x.to(device)
    edge_index = full_data.edge_index.to(device)

    encoder = build_encoder(
        in_dim=full_data.num_features,
        hid_dims=model_args.hid_dims,
        kind=model_args.encoder,
        act_fn=F.elu,
    ).to(device)

    loss_fn = LeJEPALoss(
        lambd=model_args.lambd,
        num_slices=model_args.num_slices,
    ).to(device)

    optim = AdamW(
        encoder.parameters(), lr=model_args.lr, weight_decay=model_args.weight_decay
    )

    # Recommended from LeJEPA paper.
    # scheduler = CosineAnnealingWarmRestarts(
    #     optimizer=optim, T_0=20
    # )  # TODO: Paramaterize this.

    logging.info(
        'Encoder params: %.2fM',
        sum(p.numel() for p in encoder.parameters()) / 1e6,
    )

    model_args.eval_frequency
    model_args.eval_repeat
    best_loss: float = float('inf')
    for epoch in tqdm(range(model_args.epochs), desc='Training steps'):
        encoder.train()
        prepared_graph = prepare_subgraph(
            data=full_data,
        )
        global_views, local_views, B, V_g, V_l = flatten_views([prepared_graph])

        global_views = [g.to(device) for g in global_views]
        local_views = [l.to(device) for l in local_views]

        z_global, z_local = build_views(
            encoder=encoder,
            global_views=global_views,
            local_views=local_views,
            B=B,
            V_g=V_g,
            V_l=V_l,
            edge_drop_rate=model_args.edge_drop_rate,
            feat_mask_rate=model_args.feat_mask_rate,
        )

        out = loss_fn(z_global, z_local)

        optim.zero_grad(
            set_to_none=True
        )  # Is set to none correct here? Should it be set to zero by default?
        out.total.backward()
        optim.step()
        # scheduler.step()

        if out.total < best_loss:
            save_checkpoint(
                save_dir, epoch, encoder, optim, out.total, out.pred, out.sigreg
            )
            logging.info(
                f'[step {epoch + 1}] New best loss {out.total:4f} saved best.pt'
            )
            best_loss = out.total
        if epoch % model_args.log_frequency == 0 or epoch == model_args.epochs - 1:
            msg = (
                f'epoch={epoch} loss={out.total.item():.4f} '
                f'inv={out.pred.item():.4f} uni={out.sigreg.item():.4f}'
            )
            if collapse_diagnostics is not None:
                with torch.no_grad():
                    stats = collapse_diagnostics(
                        z_global.reshape(-1, z_global.size(-1))
                    )
                msg += (
                    f' | eff_rank={stats.eff_rank:.1f} trace={stats.trace_cov:.3f} '
                    f'var[min/mean/max]={stats.var_min:.3f}/{stats.var_mean:.3f}/{stats.var_max:.3f}'
                )
            logging.info(msg)

        if (
            model_args.eval_frequency > 0
            and epoch > 0
            and epoch % model_args.eval_frequency == 0
        ):
            logging.info('epoch=%d intermediate probe...', epoch)
            embeddings = encode_full_graph(encoder, x, edge_index)
            run_eval(
                embeddings, full_data, model_args, data_args, meta_args, which=eval_mode
            )

    logging.info('Final evaluation (%s protocol)', eval_mode)
    embeddings = encode_full_graph(encoder, x, edge_index)
    run_eval(embeddings, full_data, model_args, data_args, meta_args, which=eval_mode)


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
            eval_mode=args.eval,
        )


if __name__ == '__main__':
    main()
