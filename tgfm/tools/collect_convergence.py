"""Compare convergence speed across runs from the CONV lines main.py writes.

    python tools/collect_convergence.py logs/*_conv_lr*.log
    python tools/collect_convergence.py logs/*_conv_lr*.log --metric r1_g2t --clock step

Per run: best value of the metric and when it was reached, time to 90% / 95%
of the run's own best, and the area under the curve normalised by a budget
every run completed. Across arms: time for each arm's best run to reach a
common threshold, a fraction of the lower of the two arms' bests, so both
arms are guaranteed to cross it.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

KV = re.compile(r'(\S+?)=(\S+)')
CONV = re.compile(r'\bCONV (.*)$')
CFG = re.compile(r'TIMING-CONFIG: (.*)$')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('logs', nargs='+')
parser.add_argument('--metric', default='zeroshot_macro')
parser.add_argument(
    '--clock', default='train_s', choices=['train_s', 'step', 'samples']
)
parser.add_argument('--common-frac', type=float, default=0.95)


def read_run(path: Path) -> Tuple[Dict[str, str], List[Dict[str, float]]]:
    """Config and eval points of the LAST run in the log (logs are appended)."""
    cfg: Dict[str, str] = {}
    points: List[Dict[str, float]] = []
    for line in path.read_text().splitlines():
        m = CFG.search(line)
        if m:
            cfg, points = dict(KV.findall(m.group(1))), []
            continue
        m = CONV.search(line)
        if m:
            points.append({k: float(v) for k, v in KV.findall(m.group(1))})
    return cfg, points


def first_reach(
    pts: List[Dict[str, float]], metric: str, clock: str, thr: float
) -> Optional[float]:
    for p in pts:
        if p[metric] >= thr:
            return p[clock]
    return None


def norm_auc(
    pts: List[Dict[str, float]], metric: str, clock: str, budget: float
) -> float:
    """Trapezoid area of metric vs clock on [0, budget], divided by budget."""
    area = 0.0
    for a, b in zip(pts, pts[1:]):
        x0, x1, y0, y1 = a[clock], b[clock], a[metric], b[metric]
        if x0 >= budget:
            break
        if x1 > budget:
            y1 = y0 + (y1 - y0) * (budget - x0) / (x1 - x0)
            x1 = budget
        area += (x1 - x0) * (y0 + y1) / 2
    return area / budget


def fmt(v: Optional[float]) -> str:
    return '-' if v is None else f'{v:.1f}'


def main() -> None:
    args = parser.parse_args()
    runs = []
    for p in args.logs:
        cfg, pts = read_run(Path(p))
        if not pts or args.metric not in pts[0]:
            print(f'skip {p}: no CONV points with {args.metric}')
            continue
        runs.append(
            {
                'name': Path(p).stem,
                'arm': cfg.get('arm', '?'),
                'lr': cfg.get('lr', '?'),
                'pts': pts,
            }
        )
    if not runs:
        return

    budget = min(r['pts'][-1][args.clock] for r in runs)
    print(f'metric={args.metric} clock={args.clock} common budget={budget:.1f}\n')
    print(
        f'{"run":34s} {"arm":10s} {"lr":>8s} {"best":>7s} {"@clock":>9s} '
        f'{"t90":>9s} {"t95":>9s} {"AUC":>7s}'
    )
    for r in runs:
        pts = r['pts']
        best_p = max(pts, key=lambda p: p[args.metric])
        best = best_p[args.metric]
        r['best'] = best
        print(
            f'{r["name"]:34s} {r["arm"]:10s} {r["lr"]:>8s} {best:7.4f} '
            f'{best_p[args.clock]:9.1f} '
            f'{fmt(first_reach(pts, args.metric, args.clock, 0.90 * best)):>9s} '
            f'{fmt(first_reach(pts, args.metric, args.clock, 0.95 * best)):>9s} '
            f'{norm_auc(pts, args.metric, args.clock, budget):7.4f}'
        )

    per_arm: Dict[str, dict] = {}
    for r in runs:
        if r['arm'] not in per_arm or r['best'] > per_arm[r['arm']]['best']:
            per_arm[r['arm']] = r
    if len(per_arm) < 2:
        return
    thr = args.common_frac * min(r['best'] for r in per_arm.values())
    print(f'\ncommon threshold = {args.common_frac} x min(arm bests) = {thr:.4f}')
    for arm, r in per_arm.items():
        t = first_reach(r['pts'], args.metric, args.clock, thr)
        print(
            f'  {arm:10s} (best run {r["name"]}, lr={r["lr"]}): '
            f'reaches it at {args.clock}={fmt(t)}'
        )


if __name__ == '__main__':
    main()
