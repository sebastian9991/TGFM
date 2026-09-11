"""Linear-probe evaluation of a frozen encoder (GramJEPA or GraphCLIPMM) on MM-Graph NC targets.

Cross-dataset transfer: the model is pretrained on the LP graphs
(sports-copurchase, cloth-copurchase, books-lp) and evaluated here on the two
NC graphs (ele-fashion, books-nc). The encoder is never trained on the targets.

Protocol (LeVLJEPA Sec. 5.2, transplanted to the graph embedding): freeze the
encoder, embed every node's ego-subgraph once to Z in R^{N x d}, then fit a
single linear layer Z[train] -> label. The dataset's own split.pt gives the
train/val/test node masks; nodes with the missing-label sentinel are dropped
from all three. Repeated over data_args.eval_seeds (probe re-init only, since
the embedding is deterministic).

Model selection uses validation only, at two levels:
    probe      the head is evaluated on val after every epoch and the
               best-val-accuracy epoch is the one scored on test;
    pretrain   mm_main keeps the pretraining epoch with the highest
               macro val accuracy over targets and reports its test numbers.
Test labels never influence either choice.

Metrics: accuracy and macro-F1. Macro-F1 averages per-class F1 over the labels
present in y_true or y_pred, matching sklearn's f1_score(average='macro')
default; ele-fashion's empty class id is therefore excluded rather than
counted as F1 = 0.

The graph tower's node features follow model_args.graph_feat
(tgfm.utils.mm_features), applied exactly as in pretraining.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader

from tgfm.dataset.evaluation.mm_load import load_mm_data
from tgfm.evaluation.graphclip_mm_adapter import GraphCLIPMM
from tgfm.models.legtjepa import LeGTJEPA
from tgfm.utils.args import LeGTJEPAArguments, parse_args
from tgfm.utils.logger import setup_logging
from tgfm.utils.mm_features import check_graph_features, select_graph_features
from tgfm.utils.mm_sampler import parse_mm_target_data
from tgfm.utils.path import get_root_dir

torch.backends.mha.set_fastpath_enabled(False)

# Despite the filename, neither NC dataset carries a negative sentinel:
# ele-fashion labels run 0..11 (11 of 12 ids populated), books-nc 0..10.
# The guard is kept as a cheap safeguard for other splits/bundles.
MISSING_LABEL = -1

METRIC_KEYS = ('val/acc', 'val/f1', 'test/acc', 'test/f1')

parser = argparse.ArgumentParser(
    description='Linear-probe NC evaluation of frozen LeGTJEPA on MM-Graph.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file', type=str, required=True, help='Path to configuration file.'
)
parser.add_argument(
    '--ckpt-name',
    type=str,
    default='legtjepa_best.pt',
    help='Checkpoint under weights/<experiment>/. legtjepa_best.pt is the '
    'validation-selected epoch; legtjepa.pt is the last epoch.',
)


@torch.no_grad()
def embed_all_nodes(
    model: torch.nn.Module,
    graphs: List,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    """Frozen graph embedding for every node's ego-subgraph. Z in R^{N x d}.

    Restores the caller's training mode on exit -- without this, an in-loop
    probe silently leaves BatchNorm in inference mode and dropout off for the
    remainder of training.
    """
    was_training = model.training
    model.eval()
    try:
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
        out = []
        for batch in loader:
            batch = batch.to(device)
            out.append(model.encode_graph(batch).cpu())
        return torch.cat(out, dim=0)
    finally:
        if was_training:
            model.train()


def classification_metrics(
    pred: Tensor, y: Tensor, num_classes: int
) -> Tuple[float, float]:
    """(accuracy, macro-F1). F1_c = 2 TP_c / (|true = c| + |pred = c|)."""
    acc = float((pred == y).float().mean())
    cm = torch.bincount(y * num_classes + pred, minlength=num_classes**2).view(
        num_classes, num_classes
    )
    tp = cm.diag().float()
    denom = (cm.sum(1) + cm.sum(0)).float()
    present = denom > 0
    f1 = 2 * tp[present] / denom[present]
    return acc, float(f1.mean())


def fit_linear_probe(
    z: Dict[str, Tensor],
    y: Dict[str, Tensor],
    num_classes: int,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
) -> Dict[str, float]:
    """Standardized-feature linear probe (LeVLJEPA App. C).

    z / y are keyed by 'train', 'val', 'test'. Standardization statistics come
    from train only. Returns val and test acc / macro-F1 at the epoch with the
    highest val accuracy (first occurrence on ties).
    """
    mu = z['train'].mean(0, keepdim=True)
    sd = z['train'].std(0, keepdim=True).clamp_min(1e-8)
    zs = {k: ((v - mu) / sd).to(device) for k, v in z.items()}
    ys = {k: v.to(device) for k, v in y.items()}

    probe = torch.nn.Linear(zs['train'].size(1), num_classes, bias=True).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    best: Dict[str, float] = {}
    best_val_acc = -1.0
    for _ in range(epochs):
        probe.train()
        opt.zero_grad()
        loss_fn(probe(zs['train']), ys['train']).backward()
        opt.step()

        probe.eval()
        with torch.no_grad():
            val_acc, val_f1 = classification_metrics(
                probe(zs['val']).argmax(1), ys['val'], num_classes
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc, test_f1 = classification_metrics(
                    probe(zs['test']).argmax(1), ys['test'], num_classes
                )
                best = {
                    'val/acc': val_acc,
                    'val/f1': val_f1,
                    'test/acc': test_acc,
                    'test/f1': test_f1,
                }
    return best


def evaluate_dataset(
    model: torch.nn.Module,
    name: str,
    model_args: LeGTJEPAArguments,
    seeds: List[int],
    eval_batch_size: int,
    device: torch.device,
    feat_name: str,
) -> Dict[str, Tuple[float, float]]:
    """Returns {metric_key: (mean, std) over seeds} for METRIC_KEYS."""
    was_training = model.training
    data, _, _ = load_mm_data(name, feat_name=feat_name)
    if not hasattr(data, 'val_mask'):
        raise AttributeError(
            f'{name}: load_mm_data produced no val_mask; validation selection '
            'needs split.pt val_idx attached as data.val_mask.'
        )
    data = select_graph_features(data, model_args)
    graphs = parse_mm_target_data(name, data)
    check_graph_features(graphs, data)

    # Encoder is frozen and deterministic: embed once, reuse across seeds.
    z_all = embed_all_nodes(model, graphs, device, eval_batch_size)
    y_all = data.y

    valid = y_all > MISSING_LABEL
    num_classes = int(y_all[valid].max().item()) + 1
    masks = {
        'train': data.train_mask & valid,
        'val': data.val_mask & valid,
        'test': data.test_mask & valid,
    }
    z = {k: z_all[m] for k, m in masks.items()}
    y = {k: y_all[m] for k, m in masks.items()}

    runs: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
    for seed in seeds:
        seed_everything(seed)  # varies probe init only
        res = fit_linear_probe(z, y, num_classes, device)
        for k in METRIC_KEYS:
            runs[k].append(res[k])

    if was_training:
        model.train()
    out = {}
    for k, vals in runs.items():
        t = torch.tensor(vals)
        out[k] = (float(t.mean()), float(t.std()) if len(vals) > 1 else 0.0)
    return out


def main() -> None:
    root = get_root_dir()
    args = parser.parse_args()
    meta_args, experiment_args = parse_args(root / args.config_file)
    for experiment, experiment_arg in experiment_args.exp_args.items():
        model_args = experiment_arg.model_args
        data_args = experiment_arg.data_args
        assert isinstance(model_args, LeGTJEPAArguments)
        setup_logging(meta_args.log_file_path)

        device = torch.device(
            f'cuda:{model_args.device}' if torch.cuda.is_available() else 'cpu'
        )
        model = (
            GraphCLIPMM(model_args)
            if model_args.model == 'GraphCLIP'
            else LeGTJEPA(model_args)
        ).to(device)
        ckpt_path = str(
            Path(str(meta_args.root_dir)) / 'weights' / experiment / args.ckpt_name
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        logging.info('Loaded %s (epoch %d)', ckpt_path, ckpt['epoch'])

        feat_name = getattr(model_args, 'mm_feat_name', 't5dino')
        results = {}
        for name in data_args.target_data.split('+'):
            res = evaluate_dataset(
                model,
                name,
                model_args,
                data_args.eval_seeds,
                data_args.eval_batch_size,
                device,
                feat_name,
            )
            results[name] = res
            logging.info(
                '%s | val acc %.2f | test acc %.2f +/- %.2f | test macro-F1 '
                '%.2f +/- %.2f (graph_feat=%s, seeds=%s)',
                name,
                100 * res['val/acc'][0],
                100 * res['test/acc'][0],
                100 * res['test/acc'][1],
                100 * res['test/f1'][0],
                100 * res['test/f1'][1],
                model_args.graph_feat,
                data_args.eval_seeds,
            )
        n = len(results)
        logging.info(
            'Macro test acc %.2f | macro test F1 %.2f',
            100 * sum(r['test/acc'][0] for r in results.values()) / n,
            100 * sum(r['test/f1'][0] for r in results.values()) / n,
        )
        logging.info(
            'TABLE-ROW: '
            + ' & '.join(
                f"{100 * r['test/acc'][0]:.1f} & {100 * r['test/f1'][0]:.1f}"
                for r in results.values()
            )
        )


if __name__ == '__main__':
    main()
