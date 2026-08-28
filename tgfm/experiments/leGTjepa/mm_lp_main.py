"""Link prediction on MM-Graph with frozen LeGTJEPA node features.

Reproduces the conventional-GNN link-prediction rows of Mosaic Table 5, with
the feature-encoder axis set to our method: LeGTJEPA is pretrained on the
target graph (assumed already done; checkpoint loaded from disk), its node
features are emitted once by encode_node, and SAGE / GCN / MLP are then trained
supervised on top with a dot-product link decoder. Metrics are MRR / Hits@1 /
Hits@10, ranking each positive against the 150 shipped negatives.

    pretrain (elsewhere) : LeGTJEPA on the target LP graph
    emit                 : encode_node -> N x d' feature matrix (cached)
    train                : GNN + dot-product decoder, BCE vs sampled negatives
    eval                 : rank against target_node_neg (Mosaic Sec. 3.2)

Assumptions, stated explicitly:
  * The LeGTJEPA checkpoint exists at
        <root>/weights/<experiment>/legtjepa.pt
    and was pretrained on this same target graph.
  * The GNN trains on the FULL target graph (message passing over the real
    train edge set), with the cached frozen features as data.x -- not over
    ego-subgraphs. Emission is the only per-node-subgraph step.
  * Training negatives are sampled uniformly each epoch (structural negatives);
    only the EVAL negatives are the fixed 150 shipped in the split, so the
    reported metrics match Mosaic. Training-time negative sampling matches the
    conventional-GNN recipe (Mosaic Sec. 4.1); it does not touch eval.
  * Single-GPU: feature emission and a frozen-feature GNN both fit one card.
    (Pretraining is the distributed part and runs separately, in mm_main.py.)

Run:
    python tgfm/experiments/legtjepa/mm_lp_main.py --config-file configs/lp_mmgraph.yaml
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import wandb
from torch import Tensor
from torch_geometric.utils import to_undirected

from tgfm.dataset.evaluation.mm_load import _mm_root, load_mm_data
from tgfm.models.legtjepa import LeGTJEPA
from tgfm.models.mm_models.mm_lp_models import build_encoder, decode_link
from tgfm.utils.args import LeGTJEPAArguments, parse_args
from tgfm.utils.emit_features import emit_or_load
from tgfm.utils.evaluation_utils import evaluate_split
from tgfm.utils.logger import setup_logging
from tgfm.utils.path import get_root_dir
from tgfm.utils.seed import seed_everything

torch.backends.mha.set_fastpath_enabled(False)

parser = argparse.ArgumentParser(
    description='MM-Graph link prediction on frozen LeGTJEPA features.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file', type=str, required=True, help='Path to configuration file.'
)


def load_lp_split(dataset: str) -> Dict[str, Dict[str, Tensor]]:
    lp = torch.load(_mm_root() / dataset / 'lp-edge-split.pt', map_location='cpu')
    return lp


def train_edge_index(split: Dict, num_nodes: int) -> Tensor:
    """Undirected message-passing graph from the train positives."""
    src = torch.as_tensor(split['train']['source_node']).long()
    dst = torch.as_tensor(split['train']['target_node']).long()
    return to_undirected(torch.stack([src, dst]), num_nodes=num_nodes)


def train_one_epoch(
    encoder: torch.nn.Module,
    x: Tensor,
    edge_index: Tensor,
    pos_edges: Tensor,
    optimizer: torch.optim.Optimizer,
    num_nodes: int,
    batch_size: int,
    device: torch.device,
) -> float:
    encoder.train()
    perm = torch.randperm(pos_edges.size(1), device=device)
    total, n = 0.0, 0
    for i in range(0, perm.numel(), batch_size):
        idx = perm[i : i + batch_size]
        pos = pos_edges[:, idx]
        # uniform structural negatives: corrupt the target endpoint
        neg_dst = torch.randint(0, num_nodes, (idx.numel(),), device=device)
        neg = torch.stack([pos[0], neg_dst])

        optimizer.zero_grad(set_to_none=True)
        z = encoder(x, edge_index)
        pos_score = decode_link(z, pos)
        neg_score = decode_link(z, neg)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.cat([pos_score, neg_score]),
            torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]),
        )
        loss.backward()
        optimizer.step()
        total += loss.item() * idx.numel()
        n += idx.numel()
    return total / max(1, n)


@torch.no_grad()
def eval_split(
    encoder: torch.nn.Module,
    x: Tensor,
    edge_index: Tensor,
    split: Dict,
    which: str,
    device: torch.device,
) -> Dict[str, float]:
    encoder.eval()
    z = encoder(x, edge_index)
    return evaluate_split(
        z,
        torch.as_tensor(split[which]['source_node']).long(),
        torch.as_tensor(split[which]['target_node']).long(),
        torch.as_tensor(split[which]['target_node_neg']).long(),
        decode_link,
    )


def run_lp_for_encoder(
    enc_name: str,
    x: Tensor,
    edge_index: Tensor,
    split: Dict,
    num_nodes: int,
    model_args: LeGTJEPAArguments,
    device: torch.device,
    verbose: bool,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Train one encoder, select on valid MRR, report the matching test."""
    encoder = build_encoder(
        enc_name,
        in_dim=x.size(1),
        hidden_dim=model_args.lp_hidden_dim,
        out_dim=model_args.lp_out_dim,
        num_layers=model_args.lp_num_layers,
        dropout=model_args.lp_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=model_args.lp_lr,
        weight_decay=model_args.lp_weight_decay,
    )
    pos_edges = torch.stack(
        [
            torch.as_tensor(split['train']['source_node']).long(),
            torch.as_tensor(split['train']['target_node']).long(),
        ]
    ).to(device)

    best_val_mrr, best_test, best_epoch = -1.0, {}, 0
    for epoch in range(1, model_args.lp_epochs + 1):
        loss = train_one_epoch(
            encoder,
            x,
            edge_index,
            pos_edges,
            optimizer,
            num_nodes,
            model_args.lp_batch_size,
            device,
        )
        if epoch % model_args.lp_eval_every == 0 or epoch == model_args.lp_epochs:
            val = eval_split(encoder, x, edge_index, split, 'valid', device)
            if val['mrr'] > best_val_mrr:
                best_val_mrr = val['mrr']
                best_test = eval_split(encoder, x, edge_index, split, 'test', device)
                best_epoch = epoch
            logging.info(
                '[%s ep %d] loss=%.4f val-MRR=%.4f (best val-MRR=%.4f @%d)',
                enc_name,
                epoch,
                loss,
                val['mrr'],
                best_val_mrr,
                best_epoch,
            )
            if verbose:
                wandb.log(
                    {
                        f'{enc_name}/loss': loss,
                        f'{enc_name}/val_mrr': val['mrr'],
                        f'{enc_name}/epoch': epoch,
                    }
                )
    return best_test, {'val_mrr': best_val_mrr, 'epoch': best_epoch}


def main() -> None:
    root = get_root_dir()
    args = parser.parse_args()
    meta_args, experiment_args = parse_args(root / args.config_file)

    for experiment, experiment_arg in experiment_args.exp_args.items():
        model_args = experiment_arg.model_args
        data_args = experiment_arg.data_args
        assert isinstance(model_args, LeGTJEPAArguments)
        setup_logging(meta_args.log_file_path)
        seed_everything(meta_args.global_seed)

        device = torch.device(
            f'cuda:{getattr(model_args, "device", 0)}'
            if torch.cuda.is_available()
            else 'cpu'
        )

        verbose = meta_args.verbose
        if verbose:
            mode = 'offline' if getattr(meta_args, 'wandb_offline', True) else 'online'
            wandb.init(
                project=getattr(meta_args, 'wandb_project', 'legtjepa-lp'),
                config={'experiment': experiment},
                mode=mode,
            )

        # Frozen pretrained encoder (assumed already trained on the target graph).
        root_dir = Path(str(meta_args.root_dir))
        ckpt_path = root_dir / 'weights' / experiment / 'legtjepa.pt'
        model = LeGTJEPA(model_args).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        logging.info('Loaded %s (epoch %d)', ckpt_path, ckpt['epoch'])

        emit_mode = getattr(model_args, 'emit_mode', 'node')
        cache_dir = root_dir / 'lp_features'

        for dataset in data_args.target_data.split('+'):
            logging.info('=== %s (emit_mode=%s) ===', dataset, emit_mode)
            data, _, _ = load_mm_data(dataset, feat_name=model_args.mm_feat_name)
            num_nodes = int(data.x.size(0))
            split = load_lp_split(dataset)

            x = emit_or_load(
                model,
                dataset,
                model_args,
                emit_mode,
                device,
                cache_dir,
                data_args.eval_batch_size,
            ).to(device)
            edge_index = train_edge_index(split, num_nodes).to(device)

            for enc_name in model_args.lp_encoders.split('+'):
                test, sel = run_lp_for_encoder(
                    enc_name,
                    x,
                    edge_index,
                    split,
                    num_nodes,
                    model_args,
                    device,
                    verbose,
                )
                logging.info(
                    'RESULT %s | %s | emit=%s || MRR %.4f  Hits@1 %.4f  Hits@10 %.4f '
                    '(val-MRR %.4f @ep %d)',
                    dataset,
                    enc_name,
                    emit_mode,
                    test['mrr'],
                    test['hits@1'],
                    test['hits@10'],
                    sel['val_mrr'],
                    sel['epoch'],
                )
                if verbose:
                    wandb.log(
                        {f'test/{dataset}/{enc_name}/{k}': v for k, v in test.items()}
                    )


if __name__ == '__main__':
    main()
