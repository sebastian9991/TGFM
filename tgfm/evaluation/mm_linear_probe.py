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

--representation picks which layer is probed:
    projection  graph_projection / GPS mlp output -- what the objective acts on
                and what encode_graph returns (the default, and what every
                earlier number in this project used)
    backbone    [mean-pool || center] before that head, 2*graph_hidden_dim wide
    both        both, from a single ego-subgraph pass
A head trained by the objective is free to discard whatever the objective does
not need, so it can probe worse than the layer beneath it (SimCLR Sec. 4.2).
'both' shares the subgraph sampling, which is the whole cost on books-nc, so it
is nearly free relative to running the probe twice.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    '--representation',
    type=str,
    default='projection',
    choices=['projection', 'backbone', 'both'],
    help='Which layer to probe. See the module docstring.',
)
parser.add_argument(
    '--ckpt-name',
    type=str,
    default='legtjepa_best.pt',
    help='Checkpoint under weights/<experiment>/. legtjepa_best.pt is the '
    'validation-selected epoch; legtjepa.pt is the last epoch.',
)


def batch_representations(
    model: torch.nn.Module, batch, wanted: Sequence[str]
) -> Dict[str, Tensor]:
    """{representation: (B, d)} for one batch, from a single forward pass.

    Falls back to encode_graph for models without graph_representations (the
    released-checkpoint GraphCLIPAdapter), which can only serve 'projection'.
    """
    if hasattr(model, 'graph_representations'):
        out = model.graph_representations(batch)
        missing = [name for name in wanted if name not in out]
        if missing:
            raise KeyError(f'{type(model).__name__} has no representation {missing}')
        return {name: out[name] for name in wanted}
    if tuple(wanted) != ('projection',):
        raise KeyError(
            f'{type(model).__name__} exposes only the projection output; '
            f'cannot probe {list(wanted)}.'
        )
    return {'projection': model.encode_graph(batch)}


@torch.no_grad()
def embed_all_nodes(
    model: torch.nn.Module,
    graphs: List,
    device: torch.device,
    batch_size: int,
    representations: Sequence[str] = ('projection',),
) -> Dict[str, Tensor]:
    """Frozen graph embeddings for every node's ego-subgraph, per representation.

    Restores the caller's training mode on exit -- without this, an in-loop
    probe silently leaves BatchNorm in inference mode and dropout off for the
    remainder of training.
    """
    was_training = model.training
    model.eval()
    try:
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
        chunks: Dict[str, List[Tensor]] = {name: [] for name in representations}
        for batch in loader:
            batch = batch.to(device)
            for name, z in batch_representations(model, batch, representations).items():
                chunks[name].append(z.cpu())
        return {name: torch.cat(parts, dim=0) for name, parts in chunks.items()}
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


def evaluate_dataset_reprs(
    model: torch.nn.Module,
    name: str,
    model_args: LeGTJEPAArguments,
    seeds: List[int],
    eval_batch_size: int,
    device: torch.device,
    feat_name: str,
    representations: Sequence[str] = ('projection',),
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """{representation: {metric_key: (mean, std) over seeds}} for METRIC_KEYS."""
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

    # Encoder is frozen and deterministic: embed once, reuse across seeds and
    # across representations.
    z_by_repr = embed_all_nodes(
        model, graphs, device, eval_batch_size, representations
    )
    y_all = data.y

    valid = y_all > MISSING_LABEL
    num_classes = int(y_all[valid].max().item()) + 1
    masks = {
        'train': data.train_mask & valid,
        'val': data.val_mask & valid,
        'test': data.test_mask & valid,
    }
    y = {k: y_all[m] for k, m in masks.items()}

    results: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for repr_name in representations:
        z_all = z_by_repr[repr_name]
        z = {k: z_all[m] for k, m in masks.items()}
        runs: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
        for seed in seeds:
            seed_everything(seed)  # varies probe init only
            res = fit_linear_probe(z, y, num_classes, device)
            for k in METRIC_KEYS:
                runs[k].append(res[k])
        agg = {}
        for k, vals in runs.items():
            t = torch.tensor(vals)
            agg[k] = (float(t.mean()), float(t.std()) if len(vals) > 1 else 0.0)
        agg['dim'] = (float(z_all.size(1)), 0.0)
        results[repr_name] = agg
        del z

    if was_training:
        model.train()
    return results


def evaluate_dataset(
    model: torch.nn.Module,
    name: str,
    model_args: LeGTJEPAArguments,
    seeds: List[int],
    eval_batch_size: int,
    device: torch.device,
    feat_name: str,
    representation: str = 'projection',
) -> Dict[str, Tuple[float, float]]:
    """Single-representation view, for mm_main's in-loop epoch selection.

    Return shape is exactly METRIC_KEYS, unchanged from before this function
    took a representation, so the training loop's selection metric keeps its
    meaning; only which layer it reads can change.
    """
    res = evaluate_dataset_reprs(
        model,
        name,
        model_args,
        seeds,
        eval_batch_size,
        device,
        feat_name,
        (representation,),
    )[representation]
    # 'dim' is a report field, not a metric: dropping it keeps this return
    # exactly METRIC_KEYS, which is what mm_main's wandb logging iterates over.
    return {k: v for k, v in res.items() if k in METRIC_KEYS}


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
        representations = (
            ('backbone', 'projection')
            if args.representation == 'both'
            else (args.representation,)
        )
        results = {}
        for name in data_args.target_data.split('+'):
            res = evaluate_dataset_reprs(
                model,
                name,
                model_args,
                data_args.eval_seeds,
                data_args.eval_batch_size,
                device,
                feat_name,
                representations,
            )
            results[name] = res
            for repr_name in representations:
                r = res[repr_name]
                logging.info(
                    '%s | repr=%-10s dim=%4d | VAL-ACC: %.5f | '
                    'TEST-ACC: %.5f+/-%.5f | TEST-F1: %.5f+/-%.5f',
                    name,
                    repr_name,
                    int(r['dim'][0]),
                    r['val/acc'][0],
                    *r['test/acc'],
                    *r['test/f1'],
                )
        n = len(results)
        logging.info('--------------------------------')
        for repr_name in representations:
            logging.info(
                'repr=%-10s | macro test acc %.2f | macro test F1 %.2f',
                repr_name,
                100 * sum(r[repr_name]['test/acc'][0] for r in results.values()) / n,
                100 * sum(r[repr_name]['test/f1'][0] for r in results.values()) / n,
            )
            logging.info(
                'TABLE-ROW %s/%s: %s',
                experiment,
                repr_name,
                ' & '.join(
                    f"{100 * r[repr_name]['test/acc'][0]:.1f} & "
                    f"{100 * r[repr_name]['test/f1'][0]:.1f}"
                    for r in results.values()
                ),
            )
        if len(representations) == 2:
            delta = sum(
                r['backbone']['test/acc'][0] - r['projection']['test/acc'][0]
                for r in results.values()
            ) / n
            logging.info(
                'backbone - projection, macro test acc: %+.2f points', 100 * delta
            )


if __name__ == '__main__':
    main()
