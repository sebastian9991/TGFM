"""Summarize the graph-prior regularizer runs (launches/graph_prior/graph_prior.sh).

Reads logs/gprior_<arm>_s<seed>_{nc,lp}.log and
<scratch>/graph_clip_datasets/weights/gprior_<arm>_s<seed>/train_stats.json and
prints one row per arm: mean +/- std over training seeds of zero-shot node
classification accuracy and link-prediction AUC per target, plus the batch
graph stats (rank 0, mean over the last epoch), effective rank of the graph
embeddings (RankMe) and loss forward time.

    uv run python scripts/graph_prior_summary.py [--logs logs] [--weights DIR]
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from tgfm.utils.path import get_root_dir, get_scratch

NC = re.compile(r'\] (\w+): ([\d.]+) \+/- ([\d.]+) \(direction')
LP = re.compile(r'\] (\w+) AUC: ([\d.]+) \+/- ([\d.]+) \(direction')
KEY = re.compile(r'gprior_(.+)_s(\d+)_(nc|lp)\.log$')
ARMS = ['sigreg_rand', 'sigreg_clust', 'gsig_t0.1', 'gsig_t1', 'gsig_t5', 'gmmd_t1']
TARGETS = ['cora', 'wikics', 'history']
STATS = [
    ('batch_frac_nonisolated', 'frac. centres w/ in-batch nbr', '{:.3f}'),
    ('batch_edges', 'edges / batch', '{:.1f}'),
    ('rankme_g', 'RankMe(z_g)', '{:.1f}'),
    ('loss_ms', 'loss fwd ms', '{:.1f}'),
    ('adj_ms', 'adj build ms', '{:.1f}'),
]


def fmt(xs, f='{:.2f}'):
    if not xs:
        return '-'
    if len(xs) == 1:
        return f.format(xs[0]) + ' (n=1)'
    return f'{f.format(statistics.mean(xs))} ± {f.format(statistics.stdev(xs))} (n={len(xs)})'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', default=str(get_root_dir() / 'logs'))
    ap.add_argument(
        '--weights', default=str(get_scratch() / 'graph_clip_datasets' / 'weights')
    )
    args = ap.parse_args()

    # metric[arm][(task, target)] -> list over training seeds
    metric = defaultdict(lambda: defaultdict(list))
    stats = defaultdict(lambda: defaultdict(list))
    seeds = defaultdict(set)
    for path in sorted(Path(args.logs).glob('gprior_*_s*_*.log')):
        m = KEY.search(path.name)
        if not m:
            continue
        arm, seed, task = m.group(1), int(m.group(2)), m.group(3)
        pat = NC if task == 'nc' else LP
        for line in path.read_text().splitlines():
            hit = pat.search(line)
            if hit and hit.group(1) in TARGETS:
                metric[arm][(task, hit.group(1))].append(float(hit.group(2)))
                seeds[arm].add(seed)
    for arm in ARMS:
        for seed in range(3):
            p = Path(args.weights) / f'gprior_{arm}_s{seed}' / 'train_stats.json'
            if p.exists():
                last = json.loads(p.read_text())[-1]
                for k, _, _ in STATS:
                    if k in last:
                        stats[arm][k].append(last[k])

    head = ['arm'] + [f'NC {t}' for t in TARGETS] + [f'LP-AUC {t}' for t in TARGETS]
    print('| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    for arm in ARMS:
        row = [arm] + [fmt(metric[arm][('nc', t)]) for t in TARGETS]
        row += [fmt(metric[arm][('lp', t)]) for t in TARGETS]
        print('| ' + ' | '.join(row) + ' |')
    print()
    head = ['arm'] + [name for _, name, _ in STATS]
    print('| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    for arm in ARMS:
        print(
            '| '
            + ' | '.join([arm] + [fmt(stats[arm][k], f) for k, _, f in STATS])
            + ' |'
        )


if __name__ == '__main__':
    main()
