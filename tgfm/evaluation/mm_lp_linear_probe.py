"""Link-prediction linear probe on frozen LeGTJEPA embeddings (MM-Graph).

The node-classification analogue of this file (mm_linear_probe.py) fits a
Linear(d, num_classes) head and reports accuracy. Link prediction has no head
to fit: a link (u, v) is scored directly from the two frozen endpoint
embeddings, so this is a zero-parameter readout of the representation.

    embed every node once via encode_node  ->  Z in R^{N x d}
    score(u, v) = <z_u, z_v>  (dot)  or  cos(z_u, z_v)  (cosine)
    rank each positive's target among its 150 shipped negatives
    MRR / Hits@1 / Hits@3 / Hits@10   (Mosaic LinkPredictionEvaluator metrics)

Two scores are reported. The MM-Graph benchmark leaves link scoring to the
plugged-in model (its GNN rows dot-product a *trained* encoder's output), so
there is no single canonical raw-feature score to match; dot product is the
standard decoder and the faithful choice, while cosine isolates the angle the
volume/cross-modal alignment term actually shaped -- relevant here because
SIGReg targets a Gaussian, not a sphere, so these embeddings carry norm
structure and the two scores genuinely differ.

Pretraining is assumed done on the same target graph; the checkpoint is loaded
from <root>/weights/<experiment>/legtjepa.pt. encode_node embeds each node via
its own ego-subgraph (every node the center exactly once), matching the NC
probe and the feature-emission path.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

import torch
from torch import Tensor
from torch.nn.functional import normalize
from torch_geometric.loader import DataLoader

from tgfm.dataset.evaluation.mm_load import _mm_root, load_mm_data
from tgfm.models.legtjepa import LeGTJEPA
from tgfm.utils.args import LeGTJEPAArguments, parse_args
from tgfm.utils.logger import setup_logging
from tgfm.utils.mm_sampler import parse_mm_target_data
from tgfm.utils.path import get_root_dir
from tgfm.utils.seed import seed_everything

torch.backends.mha.set_fastpath_enabled(False)

parser = argparse.ArgumentParser(
    description='Link-prediction linear probe on frozen LeGTJEPA embeddings.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file', type=str, required=True, help='Path to configuration file.'
)


@torch.no_grad()
def embed_all_nodes(
    model: LeGTJEPA,
    dataset: str,
    feat_name: str,
    device: torch.device,
    batch_size: int,
    num_workers: int = 8,
) -> Tensor:
    """Projected per-node embedding Z in R^{N x d}, row u = node u.

    Restores the caller's training mode on exit.
    """
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
        rows = [model.encode_node(b.to(device)).cpu() for b in loader]
        return torch.cat(rows, dim=0)
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def rank_metrics(pos_scores: Tensor, neg_scores: Tensor) -> Dict[str, float]:
    """Pos (P,), neg (P, K). OGB '>=' ranking. MRR, Hits@{1,3,10}."""
    ranks = (neg_scores >= pos_scores.unsqueeze(1)).sum(dim=1) + 1
    return {
        'mrr': (1.0 / ranks.float()).mean().item(),
        'hits@1': (ranks <= 1).float().mean().item(),
        'hits@3': (ranks <= 3).float().mean().item(),
        'hits@10': (ranks <= 10).float().mean().item(),
    }


@torch.no_grad()
def score_split(
    z: Tensor,
    source: Tensor,
    target: Tensor,
    target_neg: Tensor,
    metric: str,
    device: torch.device,
    batch_size: int = 4096,
) -> Dict[str, float]:
    """Score positives and their 150 negatives, batched over positives.

    metric: 'dot' -> <z_u, z_v>; 'cosine' -> <unit z_u, unit z_v>.
    """
    zq = normalize(z, dim=-1) if metric == 'cosine' else z
    zq = zq.to(device)
    pos_all, neg_all = [], []
    for i in range(0, source.numel(), batch_size):
        s = source[i : i + batch_size].to(device)
        t = target[i : i + batch_size].to(device)
        tneg = target_neg[i : i + batch_size].to(device)  # (b, K)

        zs = zq[s]  # (b, d)
        pos = (zs * zq[t]).sum(-1)  # (b,)
        # (b, 1, d) * (b, K, d) -> (b, K)
        neg = (zs.unsqueeze(1) * zq[tneg]).sum(-1)

        pos_all.append(pos.cpu())
        neg_all.append(neg.cpu())
    return rank_metrics(torch.cat(pos_all), torch.cat(neg_all))


def load_lp_split(dataset: str) -> Dict:
    return torch.load(_mm_root() / dataset / 'lp-edge-split.pt', map_location='cpu')


def evaluate_dataset(
    model: LeGTJEPA,
    dataset: str,
    model_args: LeGTJEPAArguments,
    device: torch.device,
    eval_batch_size: int,
) -> Dict[str, Dict[str, float]]:
    """Both scores on valid and test for one dataset."""
    z = embed_all_nodes(
        model, dataset, model_args.mm_feat_name, device, eval_batch_size
    )
    split = load_lp_split(dataset)

    results: Dict[str, Dict[str, float]] = {}
    for which in ('valid', 'test'):
        s = torch.as_tensor(split[which]['source_node']).long()
        t = torch.as_tensor(split[which]['target_node']).long()
        tneg = torch.as_tensor(split[which]['target_node_neg']).long()
        for metric in ('dot', 'cosine'):
            results[f'{which}/{metric}'] = score_split(
                z, s, t, tneg, metric, device, eval_batch_size
            )
    return results


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

        model = LeGTJEPA(model_args).to(device)
        ckpt_path = (
            Path(str(meta_args.root_dir)) / 'weights' / experiment / 'legtjepa.pt'
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        logging.info('Loaded %s (epoch %d)', ckpt_path, ckpt['epoch'])

        for dataset in data_args.target_data.split('+'):
            logging.info('=== %s ===', dataset)
            res = evaluate_dataset(
                model, dataset, model_args, device, data_args.eval_batch_size
            )
            for metric in ('dot', 'cosine'):
                te = res[f'test/{metric}']
                va = res[f'valid/{metric}']
                logging.info(
                    'DATA: %s | METHOD: linear-probe-lp (%s) | '
                    'TEST MRR %.4f  H@1 %.4f  H@3 %.4f  H@10 %.4f '
                    '(valid MRR %.4f)',
                    dataset,
                    metric,
                    te['mrr'],
                    te['hits@1'],
                    te['hits@3'],
                    te['hits@10'],
                    va['mrr'],
                )


if __name__ == '__main__':
    main()
