"""Plot convergence curves from the CONV lines convergence_main.py writes.

uv run python tgfm/tools/plot_convergence.py logs/*_conv_lr*.log --out logs/convergence.png
uv run python tgfm/tools/plot_convergence.py logs/*_conv_lr*.log --clock train_s
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

KV = re.compile(r'(\S+?)=(\S+)')
CONV = re.compile(r'\bCONV (.*)$')
CFG = re.compile(r'TIMING-CONFIG: (.*)$')

parser = argparse.ArgumentParser()
parser.add_argument('logs', nargs='+')
parser.add_argument('--clock', default='step', choices=['step', 'train_s', 'samples'])
parser.add_argument(
    '--metrics', nargs='+', default=['zeroshot_macro', 'r1_g2t', 'rankme']
)
parser.add_argument('--out', default='logs/convergence.png')


def read_run(path: Path):
    cfg, pts = {}, []
    for line in path.read_text().splitlines():
        if m := CFG.search(line):
            cfg, pts = dict(KV.findall(m.group(1))), []  # last run in an appended log
        elif m := CONV.search(line):
            pts.append({k: float(v) for k, v in KV.findall(m.group(1))})
    return cfg, pts


def main() -> None:
    args = parser.parse_args()
    colors = {'GramJEPA': 'tab:blue', 'GraphCLIP': 'tab:orange'}
    styles = {'1e-05': ':', '0.0001': '-', '0.0005': '--'}
    fig, axes = plt.subplots(1, len(args.metrics), figsize=(5 * len(args.metrics), 4))
    for path in args.logs:
        cfg, pts = read_run(Path(path))
        if not pts:
            continue
        arm, lr = cfg.get('arm', '?'), cfg.get('lr', '?')
        xs = [p[args.clock] for p in pts]
        for ax, metric in zip(axes, args.metrics):
            if metric in pts[0]:
                ax.plot(
                    xs,
                    [p[metric] for p in pts],
                    marker='o',
                    ms=3,
                    color=colors.get(arm),
                    ls=styles.get(lr, '-'),
                    label=f'{arm} lr={lr}',
                )
    for ax, metric in zip(axes, args.metrics):
        ax.set_xscale('symlog', linthresh=16 if args.clock == 'step' else 20)
        ax.set_xlabel(args.clock)
        ax.set_title(metric)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
