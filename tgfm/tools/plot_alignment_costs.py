"""Two-series cost figure: volumetric + SIGReg (ours) against InfoNCE.

Reads the JSON written by bench_alignment_cost.py --out and draws only the
'volume' and 'infonce' objectives; component series (volume_only, sigreg) are
ignored even if the sweep recorded them. Styling matches emit_figure in the
benchmark so the two figures sit together in the paper.

An arm that stops short of the largest batch in the sweep ran out of memory
there; its last point is labelled OOM. The volumetric curve carries an arrow
labelled 'ours' in addition to the legend entry.

    python tgfm/tools/plot_alignment_cost.py bench.json --out fig/align_cost
    python tgfm/tools/plot_alignment_cost.py bench.json --out fig/align_cost \
        --k 2 --panels time
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# Type 42 fonts: Type 3 is rejected by arXiv and several submission systems.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

OBJS = ('volume', 'infonce')
COLOR = {'volume': '#2a78d6', 'infonce': '#eb6834'}
MARKER = {'volume': 'o', 'infonce': 's'}
LABEL = {'volume': 'volumetric + SIGReg (ours)', 'infonce': 'InfoNCE'}
REF_SLOPE = {'volume': 1.0, 'infonce': 2.0}
INK, MUTED, GRID, GRID_MINOR, SPINE = (
    '#0b0b0b',
    '#52514e',
    '#e4e4e0',
    '#f0f0ec',
    '#c8c8c3',
)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('results', help='JSON from bench_alignment_cost.py --out')
parser.add_argument(
    '--out', default='alignment_cost', help='Output stem; writes .pdf and .png.'
)
parser.add_argument(
    '--k', type=int, nargs='*', default=None, help='k values to plot (default: all).'
)
parser.add_argument(
    '--panels', nargs='+', choices=['time', 'memory'], default=['time', 'memory']
)
parser.add_argument(
    '--per-sample', action='store_true', help='Divide by B: linear becomes flat.'
)
parser.add_argument(
    '--no-guides', action='store_true', help='Omit the slope-1 / slope-2 guides.'
)
parser.add_argument(
    '--no-split', action='store_true', help='Skip the one-panel-per-file copies.'
)
parser.add_argument('--facecolor', default='white', help="'none' for transparent.")
parser.add_argument('--preview-note', default='', help=argparse.SUPPRESS)


def panel_spec(field: str, per_sample: bool) -> Tuple[str, str]:
    """(row key, y label)."""
    if field == 'time':
        return 'ms', (
            'forward + backward\n($\\mu$s / sample)'
            if per_sample
            else 'forward + backward (ms)'
        )
    return 'peak_mib', (
        'peak memory\n(KiB / sample)' if per_sample else 'peak memory (MiB)'
    )


def series(
    rows: List[Dict], obj: str, k: int, key: str, per_sample: bool
) -> List[Tuple[int, float]]:
    pts = sorted(
        (r['B'], r[key]) for r in rows if r['objective'] == obj and r['k'] == k
    )
    if per_sample:
        f = 1000.0 if key == 'ms' else 1024.0
        pts = [(b, y * f / b) for b, y in pts]
    return pts


def draw(ax, rows: List[Dict], k: int, field: str, args, show_k: bool) -> bool:
    key, ylabel = panel_spec(field, args.per_sample)
    data = {o: series(rows, o, k, key, args.per_sample) for o in OBJS}
    if not any(data.values()):
        return False
    b_max = max(r['B'] for r in rows if r['k'] == k)

    lo, hi = float('inf'), 0.0
    for o in OBJS:
        if not data[o]:
            continue
        xs, ys = zip(*data[o])
        lo, hi = min(lo, min(ys)), max(hi, max(ys))
        ax.plot(
            xs,
            ys,
            color=COLOR[o],
            marker=MARKER[o],
            markersize=5,
            linewidth=2.4 if o == 'volume' else 2.0,
            label=LABEL[o],
            zorder=3 if o == 'volume' else 2,
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(lo / 1.8, hi * 2.2)

    # OOM: the arm stopped before the largest batch in the sweep.
    for o in OBJS:
        if data[o] and data[o][-1][0] < b_max:
            ex, ey = data[o][-1]
            ax.annotate(
                'OOM',
                xy=(ex, ey),
                xytext=(6, 2),
                textcoords='offset points',
                fontsize=7,
                color=COLOR[o],
                fontweight='bold',
                annotation_clip=False,
            )

    # Arrow to our curve, anchored at its last point so it lands where the
    # contrastive arm has either overtaken it or already run out of memory.
    if data['volume']:
        vx, vy = data['volume'][-1]
        ax.annotate(
            '★ ours',
            xy=(vx, vy),
            xytext=(-6, -40),
            textcoords='offset points',
            fontsize=8,
            fontweight='bold',
            color=COLOR['volume'],
            ha='center',
            va='top',
            arrowprops=dict(
                arrowstyle='->', color=COLOR['volume'], lw=1.2, shrinkA=1, shrinkB=5
            ),
            annotation_clip=False,
        )

    if not args.no_guides:
        for o in OBJS:
            pts = data[o]
            if len(pts) < 2:
                continue
            bx, by = pts[-1]
            e = REF_SLOPE[o] - (1.0 if args.per_sample else 0.0)
            gx = [max(pts[0][0], bx / 10.0), bx]
            ax.plot(
                gx,
                [by * (x / bx) ** e for x in gx],
                color='#8c8c86',
                linewidth=0.9,
                linestyle=':',
                zorder=0,
            )

    ax.set_xlabel('batch size $B$')
    ax.set_ylabel(ylabel)
    if show_k:
        ax.annotate(
            f'$k = {k}$',
            xy=(0.03, 0.93),
            xycoords='axes fraction',
            fontsize=9,
            color=INK,
            fontweight='bold',
        )
    ax.grid(True, which='major', color=GRID, linewidth=0.8)
    ax.grid(True, which='minor', color=GRID_MINOR, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.set_facecolor(args.facecolor)
    return True


def legend(fig, ax, fontsize: float) -> None:
    h, lb = ax.get_legend_handles_labels()
    leg = fig.legend(
        h,
        lb,
        loc='outside upper center',
        ncol=2,
        frameon=False,
        fontsize=fontsize,
        labelcolor=INK,
    )
    for t in leg.get_texts():
        if t.get_text() == LABEL['volume']:
            t.set_fontweight('bold')


def save(fig, stem: str, written: List[str], note: str) -> None:
    if note:
        fig.text(
            0.5,
            0.5,
            note,
            ha='center',
            va='center',
            fontsize=20,
            color='#d0d0d0',
            alpha=0.5,
            rotation=20,
            zorder=0,
        )
    d = os.path.dirname(stem)
    if d:
        os.makedirs(d, exist_ok=True)
    for ext, kw in (('.pdf', {}), ('.png', {'dpi': 200})):
        fig.savefig(
            stem + ext,
            bbox_inches='tight',
            facecolor=fig.get_facecolor(),
            edgecolor='none',
            **kw,
        )
        written.append(stem + ext)


def main() -> None:
    args = parser.parse_args()
    with open(args.results) as fh:
        rows = [
            r
            for r in json.load(fh)['rows']
            if r['sweep'] == 'batch' and r['objective'] in OBJS
        ]
    ks = sorted({r['k'] for r in rows})
    if args.k:
        ks = [k for k in ks if k in args.k]
    if not ks:
        raise SystemExit('no batch-sweep rows for the requested k')

    written: List[str] = []
    nrow, ncol = len(ks), len(args.panels)
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(3.6 * ncol, 2.9 * nrow),
        constrained_layout=True,
        facecolor=args.facecolor,
        squeeze=False,
    )
    for ri, k in enumerate(ks):
        for ci, field in enumerate(args.panels):
            draw(axes[ri][ci], rows, k, field, args, show_k=(nrow > 1 and ci == 0))
    legend(fig, axes[0][0], 8)
    save(fig, args.out, written, args.preview_note)
    plt.close(fig)

    if not args.no_split and nrow * ncol > 1:
        for k in ks:
            for field in args.panels:
                sub, ax = plt.subplots(
                    figsize=(3.6, 2.9),
                    constrained_layout=True,
                    facecolor=args.facecolor,
                )
                if draw(ax, rows, k, field, args, show_k=len(ks) > 1):
                    legend(sub, ax, 7)
                    stem = (
                        f'{args.out}_k{k}_{field}'
                        if len(ks) > 1
                        else f'{args.out}_{field}'
                    )
                    save(sub, stem, written, args.preview_note)
                plt.close(sub)
    print('wrote ' + ', '.join(written))


if __name__ == '__main__':
    main()
