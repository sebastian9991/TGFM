"""Training loop for SSGE (Negative-Free Self-Supervised Gaussian Embedding).

Re-implemented to match the LeJEPA/LeGraph codebase conventions while staying
faithful to https://github.com/Cloudy1225/SSGE.

Key design points (contrast with the LeJEPA pretrainer):
    - Single shared encoder (GCN, or MLP for CoauthorCS). No EMA / teacher.
    - SSGE is *full-graph, node-level*: there is no subgraph sampling and no
      view pooling. Each "epoch" = two fresh augmentations of the whole graph,
      one batched encode of each, invariance + uniformity loss, backprop, step.
    - Augmentations: edge dropping + feature masking (node identity preserved).
    - Loss: invariance (per-node alignment) + lambda * uniformity (W2 to N(0,I)).
      Returned as SSGEOutput(total, pred, sigreg) to match the LeJEPALoss API.

Evaluation:
    --eval yours  : use the repo's evaluate_linear_probe / node_classification
                    (scores SSGE on *your* probe; recommended for baselining
                    SSGE against your own model under identical conditions).
    --eval paper  : use tgfm.dataset.evaluation.ssge_eval (the paper's exact
                    LR-probe + clustering protocol; for reproducing their
                    reported numbers).

Integration notes (verify against your actual signatures):
    - get_dataset(root: str, name: str, transform=None) -> dataset; dataset[0]
      is a PyG Data with .x, .y, .edge_index, .num_features and (for some
      datasets) train/val/test masks.
    - The "yours" eval path calls evaluate_linear_probe(embeddings, full_data,
      repeat=, data_random_seed=) exactly as in your main.py.
"""

import argparse
import logging
import random
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch_geometric.data import Data
from tqdm import tqdm

from tgfm.dataset.pyg.data import get_dataset
from tgfm.models.ssge.ssge import SSGELoss, build_encoder
from tgfm.utils.args import (
    DataArguments,
    MetaArguments,
    ModelArguments,
    SSGEArguments,
    parse_args,
)
from tgfm.utils.diagnostics import collapse_diagnostics
from tgfm.utils.logger import setup_logging
from tgfm.utils.path import get_root_dir
from tgfm.utils.seed import seed_everything
from tgfm.views.ssge.ssge_augments import augment

parser = argparse.ArgumentParser(
    description='SSGE pretraining.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file', type=str, required=True, help='Path to configuration file.'
)
parser.add_argument('--eval', type=str, default='yours', choices=['yours', 'paper'])


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    total_loss: float,
    pred_loss: float,
    sig_loss: float,
) -> None:
    """Atomic checkpoint (tmp + rename), same pattern as the LeJEPA trainer."""
    rng_states = {
        'python': random.getstate(),
        'torch_cpu': torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_states['torch_cuda_all'] = torch.cuda.random.get_rng_state_all()
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rng_states': rng_states,
        'total_loss': total_loss,
        'pred_loss': pred_loss,
        'sig_loss': sig_loss,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp_path)
    tmp_path.rename(path)


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
    assert isinstance(model_args, SSGEArguments)
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


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    meta_args: MetaArguments,
    save_dir: Path,
    eval_mode: str = 'yours',
) -> None:
    assert isinstance(model_args, SSGEArguments)
    device = model_args.device

    dataset = get_dataset(root=str(meta_args.root_dir), name=data_args.data_name)
    full_data: Data = dataset[0]
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
    loss_fn = SSGELoss(lambd=model_args.lam, normalize=True).to(device)
    optim = Adam(
        encoder.parameters(), lr=model_args.lr, weight_decay=model_args.weight_decay
    )
    # NB: SSGE uses plain Adam with no LR scheduler.

    logging.info(
        'Encoder params: %.3fK', sum(p.numel() for p in encoder.parameters()) / 1e3
    )

    best_loss = float('inf')
    save_path = save_dir / f'ssge_{data_args.data_name}_best.pt'

    for epoch in tqdm(range(model_args.epochs), desc='SSGE epochs'):
        encoder.train()

        x1, ei1 = augment(
            x, edge_index, model_args.edge_drop_rate, model_args.feat_mask_rate
        )
        x2, ei2 = augment(
            x, edge_index, model_args.edge_drop_rate, model_args.feat_mask_rate
        )

        z1 = encoder(x1, ei1)
        z2 = encoder(x2, ei2)

        out = loss_fn(z1, z2)

        optim.zero_grad(set_to_none=True)
        out.total.backward()
        optim.step()

        if out.total.item() < best_loss:
            best_loss = out.total.item()
            save_checkpoint(
                save_path,
                epoch,
                encoder,
                optim,
                out.total.item(),
                out.pred.item(),
                out.sigreg.item(),
            )

        if epoch % model_args.log_frequency == 0 or epoch == model_args.epochs - 1:
            msg = (
                f'epoch={epoch} loss={out.total.item():.4f} '
                f'inv={out.pred.item():.4f} uni={out.sigreg.item():.4f}'
            )
            if collapse_diagnostics is not None:
                with torch.no_grad():
                    stats = collapse_diagnostics(z1.detach())
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
