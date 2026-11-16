"""Per-dataset statistics for the MM-Graph / TAG targets, joined to Delta.

Produces the table that sits beside Delta = GramJEPA - GraphCLIP so the
per-dataset pattern in Tables 1-4 can be read against dataset properties
rather than asserted.

Two distinct questions need two different runs, and conflating them is the
mistake this script is built to avoid:

  --split target      properties of the EVALUATION graphs. These can explain
                      why transfer differs per target. Delta is attached.
  --split pretrain    properties of the PRETRAINING pool. This is where
                      InfoNCE draws its negatives, so the collision and
                      near-duplicate rates that bear on the negatives argument
                      are the ones measured HERE. They are a single global
                      property of the pool and therefore cannot explain
                      per-target variation in Delta.

Columns
  nodes, edges, mean degree
  edge homophily          fraction of edges whose endpoints share a label
                          (Zhu et al., NeurIPS 2020). Labelled graphs only.
  classes, H(y)           label entropy in bits, and H/log2(C) as a
                          balance measure
  collision p             sum_c p_c^2: probability two nodes drawn at random
                          share a label, i.e. the rate at which an InfoNCE
                          negative is semantically a positive
  dup@0.95                fraction of node pairs with feature cosine > tau,
                          estimated from --pair-samples random pairs
  false neg / anchor      (B-1) * max(collision p, dup rate) at --batch:
                          expected wrongly-repelled pairs per anchor per step
  mean cos                mean pairwise feature cosine; high values mean the
                          feature space is anisotropic and negatives are easy
  text chars              mean length of the node text, where available

Usage (from the repo root, so the loaders resolve):
    python tools/dataset_stats.py --split target --out analysis/target_stats.json
    python tools/dataset_stats.py --split pretrain --batch 1800
"""

import argparse
import json
import math
from typing import Dict, List

import torch

parser = argparse.ArgumentParser(
    description='Dataset statistics joined to the GramJEPA-GraphCLIP delta.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--split', choices=['target', 'pretrain'], default='target')
parser.add_argument('--datasets', type=str, default='', help="Override, '+'-separated.")
parser.add_argument(
    '--batch', type=int, default=1800, help='Global batch for the false-negative count.'
)
parser.add_argument('--dup-tau', type=float, default=0.95)
parser.add_argument('--pair-samples', type=int, default=4_000_000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--mm-feat-name', type=str, default='t5dino')
parser.add_argument('--text-dim', type=int, default=768)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--no-text', action='store_true', help='Skip the text-length pass.')
parser.add_argument('--out', type=str, default='')

TAG_TARGETS = [
    'cora',
    'citeseer',
    'wikics',
    'instagram',
    'photo',
    'computer',
    'history',
]
TAG_PRETRAIN = ['ogbn-arxiv', 'arxiv_2023', 'pubmed', 'ogbn-products', 'reddit']
MM_TARGETS = ['ele-fashion', 'books-nc']
MM_PRETRAIN = ['sports-copurchase', 'cloth-copurchase', 'books-lp']

# Delta = GramJEPA - GraphCLIP, from the paper's Tables 1-4. Positive favours
# GramJEPA. Keys are this script's dataset names.
DELTA: Dict[str, Dict[str, float]] = {
    # Table 1 zero-shot NC accuracy; Table 2 LP AUC; Table 3 LP MRR
    'cora': {'zs_nc': 65.68 - 67.31, 'lp_auc': 97.77 - 83.15, 'lp_mrr': 64.64 - 68.07},
    'citeseer': {'zs_nc': 67.40 - 63.13, 'lp_mrr': 82.81 - 84.44},
    'wikics': {
        'zs_nc': 62.32 - 70.19,
        'lp_auc': 93.90 - 92.67,
        'lp_mrr': 57.42 - 53.37,
    },
    'instagram': {'zs_nc': 65.04 - 64.05, 'lp_mrr': 41.07 - 41.06},
    'photo': {'zs_nc': 54.86 - 53.40, 'lp_mrr': 57.20 - 51.84},
    'computers': {'zs_nc': 61.14 - 62.04, 'lp_mrr': 64.20 - 59.20},
    'history': {
        'zs_nc': 48.47 - 53.88,
        'lp_auc': 98.57 - 96.04,
        'lp_mrr': 69.47 - 68.02,
    },
    # Table 4 linear probe, best modality pair per method (G-T for both)
    'ele-fashion': {'probe_gt': 83.7 - 81.6, 'probe_gi': 83.2 - 82.1},
    'books-nc': {'probe_gt': 79.1 - 75.6, 'probe_gi': 76.9 - 72.3},
}


def entropy_bits(counts: torch.Tensor) -> float:
    p = counts.double() / counts.sum()
    p = p[p > 0]
    return float(-(p * p.log2()).sum())


def collision_prob(counts: torch.Tensor) -> float:
    """sum_c p_c^2, the chance two independently drawn nodes share a label."""
    p = counts.double() / counts.sum()
    return float((p * p).sum())


def edge_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of edges joining same-label nodes (Zhu et al., 2020)."""
    src, dst = edge_index[0], edge_index[1]
    keep = (y[src] >= 0) & (y[dst] >= 0) & (src != dst)
    if int(keep.sum()) == 0:
        return float('nan')
    return float((y[src[keep]] == y[dst[keep]]).double().mean())


@torch.no_grad()
def pair_cosine_stats(
    x: torch.Tensor, tau: float, n_pairs: int, device: torch.device, seed: int
) -> Dict[str, float]:
    """Mean pairwise cosine and P(cos > tau), from uniformly sampled pairs.

    All-pairs is N^2 and books-nc alone would be 4.7e11 pairs. A uniform sample
    is an unbiased estimator of both quantities; the standard error on a rate p
    is sqrt(p(1-p)/n), reported so a rate near the resolution floor is not read
    as a measurement.
    """
    N = x.shape[0]
    g = torch.Generator().manual_seed(seed)
    i = torch.randint(0, N, (n_pairs,), generator=g)
    j = torch.randint(0, N, (n_pairs,), generator=g)
    keep = i != j
    i, j = i[keep], j[keep]
    n = i.numel()

    total, hits, chunk = 0.0, 0, 1_000_000
    for s in range(0, n, chunk):
        ii, jj = i[s : s + chunk], j[s : s + chunk]
        a = torch.nn.functional.normalize(x[ii].to(device).float(), dim=-1)
        b = torch.nn.functional.normalize(x[jj].to(device).float(), dim=-1)
        c = (a * b).sum(-1)
        total += float(c.sum())
        hits += int((c > tau).sum())
    rate = hits / n
    return {
        'mean_cos': total / n,
        'dup_rate': rate,
        'dup_se': math.sqrt(max(rate * (1 - rate), 0.0) / n),
        'pairs_used': n,
    }


def load_tag(name: str, text_dim: int, want_text: bool):
    """(x, edge_index, y, mean_text_chars) for a GraphCLIP TAG dataset."""
    from tgfm.dataset.evaluation.load import load_data

    data, text, classes, _ = load_data(name)
    chars = float('nan')
    if want_text:
        # load_data's `text` is the per-node raw text when the loader supplies
        # it; the LLM summaries GraphCLIP trains against live in the
        # target_data JSON instead, so prefer those when present.
        summaries = _load_summaries(name)
        source = summaries if summaries else text
        if source:
            chars = sum(len(t) for t in source) / len(source)
    return data.x, data.edge_index, getattr(data, 'y', None), chars, len(classes or [])


def _load_summaries(name: str) -> List[str]:
    """Per-node LLM summaries from scratch/graph_clip_datasets/target_data/<name>.json.

    Same file parse_target_data reads; it keeps the `summary` field that the
    ego-subgraph builder discards.
    """
    try:
        from tgfm.utils.path import get_scratch

        path = get_scratch() / 'graph_clip_datasets' / 'target_data' / f'{name}.json'
        if not path.is_file():
            return []
        with open(path) as fh:
            return [d['summary'] for d in json.load(fh) if 'summary' in d]
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        print(f'  ({name}: no summaries, {type(exc).__name__}: {exc})')
        return []


def load_mm(name: str, feat_name: str, text_dim: int):
    """(text_x, edge_index, y, nan, n_classes) for an MM-Graph dataset."""
    from tgfm.dataset.evaluation.mm_load import load_mm_data

    data, _, classes = load_mm_data(name, feat_name=feat_name)
    x = data.x
    if x.shape[1] > text_dim:  # bundle not split by the loader
        x = x[:, :text_dim]
    y = getattr(data, 'y', None)
    n_classes = len(classes) if classes else (int(y.max()) + 1 if y is not None else 0)
    return x, data.edge_index, y, float('nan'), n_classes


def main() -> None:
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.datasets:
        tag = [d for d in args.datasets.split('+') if d not in MM_TARGETS + MM_PRETRAIN]
        mm = [d for d in args.datasets.split('+') if d in MM_TARGETS + MM_PRETRAIN]
    elif args.split == 'target':
        tag, mm = TAG_TARGETS, MM_TARGETS
    else:
        tag, mm = TAG_PRETRAIN, MM_PRETRAIN

    rows = []
    for name in tag + mm:
        is_mm = name in MM_TARGETS + MM_PRETRAIN
        print(f'loading {name} ...')
        try:
            if is_mm:
                x, ei, y, chars, n_classes = load_mm(
                    name, args.mm_feat_name, args.text_dim
                )
            else:
                x, ei, y, chars, n_classes = load_tag(
                    name, args.text_dim, not args.no_text
                )
        except Exception as exc:  # noqa: BLE001 - one bad dataset must not stop the table
            print(f'  SKIP {name}: {type(exc).__name__}: {exc}')
            continue

        N = x.shape[0]
        E = ei.shape[1]
        row: Dict[str, object] = {
            'dataset': name,
            'nodes': N,
            'edges': E,
            'mean_degree': 2.0 * E / N,  # undirected convention: each edge two ends
            'classes': n_classes,
            'text_chars': chars,
        }
        if y is not None:
            valid = y[y >= 0]
            counts = torch.bincount(valid)
            row['entropy_bits'] = entropy_bits(counts)
            row['entropy_norm'] = (
                row['entropy_bits'] / math.log2(len(counts)) if len(counts) > 1 else 0.0
            )
            row['collision_p'] = collision_prob(counts)
            row['homophily'] = edge_homophily(ei, y)
        else:
            row.update(
                {
                    'entropy_bits': float('nan'),
                    'entropy_norm': float('nan'),
                    'collision_p': float('nan'),
                    'homophily': float('nan'),
                }
            )

        row.update(
            pair_cosine_stats(x, args.dup_tau, args.pair_samples, device, args.seed)
        )
        worst = max(
            [
                v
                for v in (row['collision_p'], row['dup_rate'])
                if isinstance(v, float) and v == v
            ]
            or [float('nan')]
        )
        row['false_neg_per_anchor'] = (args.batch - 1) * worst
        row.update({f'delta_{k}': v for k, v in DELTA.get(name, {}).items()})
        rows.append(row)

    def fmt(v, spec='.3f'):
        if v is None:
            return '-'
        if isinstance(v, float) and v != v:
            return '-'
        return format(v, spec) if isinstance(v, float) else str(v)

    print(f'\n=== {args.split} datasets (tau={args.dup_tau}, B={args.batch}) ===')
    head = (
        f'{"dataset":<18}{"nodes":>9}{"deg":>7}{"homoph":>8}{"C":>4}{"H(y)":>7}'
        f'{"coll p":>9}{"dup@t":>9}{"fn/anch":>9}{"meancos":>9}{"chars":>8}'
    )
    print(head)
    print('-' * len(head))
    for r in rows:
        print(
            f'{r["dataset"]:<18}{r["nodes"]:>9}{fmt(r["mean_degree"],".1f"):>7}'
            f'{fmt(r["homophily"],".3f"):>8}{r["classes"]:>4}'
            f'{fmt(r["entropy_bits"],".2f"):>7}{fmt(r["collision_p"],".4f"):>9}'
            f'{fmt(r["dup_rate"],".2e"):>9}{fmt(r["false_neg_per_anchor"],".2f"):>9}'
            f'{fmt(r["mean_cos"],".3f"):>9}{fmt(r["text_chars"],".0f"):>8}'
        )

    deltas = sorted({k for r in rows for k in r if k.startswith('delta_')})
    if deltas:
        print(f'\n=== Delta (GramJEPA - GraphCLIP), points ===')
        head2 = f'{"dataset":<18}' + ''.join(f'{d[6:]:>12}' for d in deltas)
        print(head2)
        print('-' * len(head2))
        for r in rows:
            print(
                f'{r["dataset"]:<18}'
                + ''.join(f'{fmt(r.get(d), "+.2f"):>12}' for d in deltas)
            )

    if args.out:
        with open(args.out, 'w') as fh:
            json.dump({'args': vars(args), 'rows': rows}, fh, indent=2)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
