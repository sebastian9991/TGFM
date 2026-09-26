"""Linear probe on raw MM-Graph features -- no pretraining, no encoder.

The floor for the transfer table. Every pretrained row has to beat this: if a
GraphCLIP or GramJEPA graph embedding does not probe better than the frozen
input feature it was built from, the pretraining added nothing the probe can
use.

Protocol is identical to mm_linear_probe (same fit_linear_probe, same
validation-selected epoch, same seeds, same acc / macro-F1), with one
difference: Z is the raw per-node feature rather than a frozen ego-subgraph
embedding. No sampler, no GPS forward, so books-nc takes seconds instead of
~50 minutes.

Feature variants (--feats), against t5dino = [T5_768 || DINOv2-base_768]:
    t5        columns  :768   the target GraphCLIP's graph head regresses onto
    dino      columns  768:   the G-I row's node feature and image target
    t5dino    all 1536         both, concatenated

--propagate k additionally probes the k-step mean-aggregated feature
S^k X with S = D^-1 A (self-loops added), the SGC/`x + neighbours` baseline.
k=0 is the raw feature. This separates "the labels are linearly readable from
the input feature" from "message passing over the NC graph helps", which is
what a pretrained graph tower has to beat to justify itself.
"""

import argparse
import logging

import torch
from torch import Tensor

from tgfm.dataset.evaluation.mm_load import load_mm_data
from tgfm.evaluation.mm_linear_probe import (
    METRIC_KEYS,
    MISSING_LABEL,
    fit_linear_probe,
)
from tgfm.utils.logger import setup_logging

parser = argparse.ArgumentParser(
    description='Linear probe on raw MM-Graph node features (no pretraining).',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--datasets', type=str, default='ele-fashion+books-nc')
parser.add_argument(
    '--feats',
    type=str,
    default='t5,dino,t5dino',
    help='Comma-separated subset of t5,dino,t5dino.',
)
parser.add_argument(
    '--propagate',
    type=str,
    default='0',
    help='Comma-separated propagation depths k. 0 = raw feature.',
)
parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
parser.add_argument('--mm-feat-name', type=str, default='t5dino')
parser.add_argument('--text-dim', type=int, default=768)
parser.add_argument('--log-file', type=str, default='mm_raw_probe.log')

SLICES = {'t5': 'text', 'dino': 'image', 't5dino': 'both'}


def select_feature(data, name: str, text_dim: int) -> Tensor:
    """Raw feature matrix for one variant, (N, d)."""
    x = data.x
    if x.size(1) == text_dim and hasattr(data, 'image_x'):
        # Loader already split the bundle: x is text, image_x is image.
        text_x, image_x = x, data.image_x
    else:
        text_x, image_x = x[:, :text_dim], x[:, text_dim:]
    if name == 't5':
        return text_x
    if name == 'dino':
        return image_x
    return torch.cat((text_x, image_x), dim=1)


def propagate(z: Tensor, edge_index: Tensor, k: int, device: torch.device) -> Tensor:
    """k steps of mean aggregation with self-loops: (D^-1 A)^k Z, A symmetrized.

    Row-normalized rather than symmetric-normalized so every node's feature
    stays on the scale of its own input, which keeps the k=0 and k>0 probes
    comparable without refitting the standardization.
    """
    if k == 0:
        return z
    n = z.size(0)
    src, dst = edge_index[0].to(device), edge_index[1].to(device)
    # Symmetrize and add self-loops; mm_load's NC edge list is one row per edge.
    loop = torch.arange(n, device=device)
    row = torch.cat((src, dst, loop))
    col = torch.cat((dst, src, loop))
    deg = torch.zeros(n, device=device).index_add_(
        0, row, torch.ones(row.numel(), device=device)
    )
    out = z.to(device)
    for _ in range(k):
        agg = torch.zeros_like(out).index_add_(0, row, out[col])
        out = agg / deg.unsqueeze(1).clamp_min(1)
    return out.cpu()


def main() -> None:
    args = parser.parse_args()
    setup_logging(args.log_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feats = [f.strip() for f in args.feats.split(',') if f.strip()]
    depths = [int(k) for k in args.propagate.split(',') if k.strip()]
    for f in feats:
        if f not in SLICES:
            raise ValueError(f'unknown feature variant {f!r}; use {list(SLICES)}')

    table = {}
    for name in args.datasets.split('+'):
        data, _, _ = load_mm_data(name, feat_name=args.mm_feat_name)
        if not hasattr(data, 'val_mask'):
            raise AttributeError(f'{name}: no val_mask; cannot select on validation.')
        y_all = data.y
        valid = y_all > MISSING_LABEL
        num_classes = int(y_all[valid].max().item()) + 1
        masks = {
            'train': data.train_mask & valid,
            'val': data.val_mask & valid,
            'test': data.test_mask & valid,
        }
        y = {k: y_all[m] for k, m in masks.items()}
        logging.info(
            '%s: %d nodes, %d labelled, %d classes, train/val/test %d/%d/%d',
            name,
            y_all.numel(),
            int(valid.sum()),
            num_classes,
            *(int(m.sum()) for m in masks.values()),
        )

        for feat in feats:
            z_raw = select_feature(data, feat, args.text_dim)
            for k in depths:
                z_all = propagate(z_raw, data.edge_index, k, device)
                z = {s: z_all[m] for s, m in masks.items()}
                runs = {m: [] for m in METRIC_KEYS}
                for seed in args.seeds:
                    torch.manual_seed(seed)
                    res = fit_linear_probe(z, y, num_classes, device)
                    for m in METRIC_KEYS:
                        runs[m].append(res[m])
                agg = {
                    m: (
                        float(torch.tensor(v).mean()),
                        float(torch.tensor(v).std()) if len(v) > 1 else 0.0,
                    )
                    for m, v in runs.items()
                }
                table[(name, feat, k)] = agg
                logging.info(
                    '%s | feat=%-7s prop=%d dim=%4d | VAL-ACC: %.5f | '
                    'TEST-ACC: %.5f+/-%.5f | TEST-F1: %.5f+/-%.5f',
                    name,
                    feat,
                    k,
                    z_all.size(1),
                    agg['val/acc'][0],
                    *agg['test/acc'],
                    *agg['test/f1'],
                )

    logging.info('--------------------------------')
    datasets = args.datasets.split('+')
    for feat in feats:
        for k in depths:
            cells = []
            for name in datasets:
                agg = table[(name, feat, k)]
                cells.append(
                    f"{100 * agg['test/acc'][0]:.1f} & {100 * agg['test/f1'][0]:.1f}"
                )
            logging.info('TABLE-ROW raw-%s-prop%d: %s', feat, k, ' & '.join(cells))


if __name__ == '__main__':
    main()
