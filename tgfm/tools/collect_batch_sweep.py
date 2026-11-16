"""Fit the end-to-end batch-size sweep and say what it does and does not show.

Input: the logs from tgfm/tools/make_batch_sweep_configs.py, one per (arm, batch
size), each carrying a TIMING-CONFIG fingerprint and one TIMING epoch line.

What this fits, and why not epoch_s. One epoch at batch B has pool/(B*W)
steps, so epoch_s falls roughly like 1/B whatever the objective does. It is
not a cost curve. The comparable quantities are per step:

    loss_ms        alignment forward. The B vs B^2 separation lives here.
    step_ms/B      per-subgraph step cost, the end-to-end claim.
    peak_mem_mib   space.

What to expect, stated before seeing the numbers so the fit is not read
backwards. step_ms is dominated by the encoder, which is linear in the
per-rank batch, so both arms should fit close to exponent 1 on step_ms and the
per-subgraph cost should be close to flat. The separation should appear in
loss_ms, where tgfm/tools/bench_alignment_cost.py measured 1.03 +/- 0.02 for the
volumetric term and 1.95 +/- 0.05 for InfoNCE above global B = 2048. If
loss_ms here fits near 2 for InfoNCE and near 1 for volume, the isolated
benchmark carries over to the real loop; if the InfoNCE exponent here is lower,
the grid did not reach the regime and the honest statement is that at the
batch sizes this encoder admits, the objective is not what costs.

    python tgfm/tools/collect_batch_sweep.py logs/bsweep_*.log
    python tgfm/tools/collect_batch_sweep.py logs/bsweep_*.log \\
        --emit-latex analysis/batch_sweep.tex --emit-figure analysis/batch_sweep
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

parser = argparse.ArgumentParser(
    description='Fit time and memory against batch size for each arm.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('logs', nargs='+', help='Run logs from the batch sweep.')
parser.add_argument(
    '--x',
    type=str,
    default='global_batch',
    choices=('global_batch', 'batch_size'),
    help='Fit against the global batch (what InfoNCE logits scale with) or the '
    'per-rank batch (what the encoder scales with).',
)
parser.add_argument(
    '--fit-from',
    type=int,
    default=0,
    help='Drop points below this x before fitting. Small batches are launch- '
    'and latency-bound and sit off the asymptote; 0 fits everything.',
)
parser.add_argument(
    '--skip-epochs',
    type=int,
    default=0,
    help='Leading epochs to drop. The sweep runs one epoch per point, so 0.',
)
parser.add_argument(
    '--min-samples',
    type=int,
    default=4,
    help='Warn when a point has fewer sampled steps than this; its medians are '
    'thin and the point should not carry a fit on its own.',
)
parser.add_argument('--emit-latex', type=str, default='')
parser.add_argument(
    '--emit-figure',
    type=str,
    default='',
    help='Path stem; writes both .pdf and .png.',
)
parser.add_argument('--out', type=str, default='')

ARM_LABEL = {
    'volume': 'GramJEPA (volume + SIGReg)',
    'infonce': 'GraphCLIP (InfoNCE)',
    'mse': 'LeGTJEPA (cross-modal MSE)',
}
ARM_COLOR = {'volume': '#1b6ca8', 'infonce': '#c1352b', 'mse': '#4c8c3f'}
ARM_MARKER = {'volume': 'o', 'infonce': 's', 'mse': '^'}
ARM_ORDER = ('volume', 'infonce', 'mse')

# Measured in tgfm/tools/bench_alignment_cost.py on synthetic embeddings, fp32,
# d=768, fast determinant. Quoted so the end-to-end fit can be read against it.
ISOLATED = {
    'volume': (1.03, 0.02),
    'infonce': (1.95, 0.05),
}
CROSSOVER = {2: 2554, 3: 1153}

CONFIG_RE = re.compile(r'TIMING-CONFIG:\s*(.+)$')
PLAN_RE = re.compile(r'TIMING-PLAN\s+steps=(\d+)')
EPOCH_RE = re.compile(
    r'TIMING epoch\s+(?P<epoch>\d+):\s*'
    r'epoch=(?P<epoch_s>[\d.]+)s\s+'
    r'steps=(?P<steps>\d+)\s+'
    r'step=(?P<step_ms>[\d.]+)ms\s+'
    r'\[data=(?P<data_ms>[\d.]+)\s+'
    r'fwd=(?P<forward_ms>[\d.]+)\s+'
    r'loss=(?P<loss_ms>[\d.]+)\s+'
    r'bwd=(?P<backward_ms>[\d.]+)\s+'
    r'opt=(?P<optim_ms>[\d.]+)\]\s+'
    r'loss_fwd_share=(?P<loss_fwd_share>[\d.]+)\s+'
    r'thru=(?P<throughput_per_s>[\d.]+)/s\s+'
    r'peak=(?P<peak_mem_mib>[\d.]+)MiB'
    r'(?:\s+samples=(?P<samples>\d+))?'
)
OOM_RE = re.compile(r'CUDA out of memory|torch\.OutOfMemoryError|out of memory', re.I)

FIELDS = (
    'epoch_s',
    'step_ms',
    'data_ms',
    'forward_ms',
    'loss_ms',
    'backward_ms',
    'optim_ms',
    'loss_fwd_share',
    'throughput_per_s',
    'peak_mem_mib',
)

# Two-sided 95% t critical values by degrees of freedom. A log-log fit over
# five or six points has few degrees of freedom and 1.96 would understate the
# interval badly.
T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    12: 2.179,
    15: 2.131,
    20: 2.086,
    30: 2.042,
}


def t_crit(df: int) -> float:
    if df <= 0:
        return float('nan')
    for k in sorted(T95):
        if df <= k:
            return T95[k]
    return 1.96


def coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {'True': True, 'False': False, 'None': None}.get(v, v)


def parse_log(path: Path, skip_epochs: int) -> Optional[Dict]:
    """One point of the sweep, or an OOM marker when the run never trained."""
    text = path.read_text(errors='replace')
    fingerprint: Dict[str, object] = {}
    epochs: List[Dict[str, float]] = []
    planned_steps = 0
    for line in text.splitlines():
        m = CONFIG_RE.search(line)
        if m:
            fingerprint = {
                k: coerce(v)
                for tok in m.group(1).split()
                if '=' in tok
                for k, v in [tok.split('=', 1)]
            }
            continue
        m = PLAN_RE.search(line)
        if m:
            planned_steps = int(m.group(1))
            continue
        m = EPOCH_RE.search(line)
        if m:
            row = {f: float(m.group(f)) for f in FIELDS}
            row['epoch'] = float(m.group('epoch'))
            row['steps'] = float(m.group('steps'))
            row['samples'] = float(m.group('samples') or 0)
            epochs.append(row)

    oom = bool(OOM_RE.search(text))
    if not epochs:
        if not fingerprint:
            return None
        # The run reached model construction and then died. If it died on
        # memory that is the measurement: this arm cannot hold this batch.
        return {
            'name': path.stem,
            'path': str(path),
            'config': fingerprint,
            'status': 'oom' if oom else 'no-timing',
            'planned_steps': planned_steps,
        }

    kept = [e for e in epochs if e['epoch'] > skip_epochs] or epochs
    agg = {f: _median([e[f] for e in kept]) for f in FIELDS}
    return {
        'name': path.stem,
        'path': str(path),
        'config': fingerprint,
        'status': 'ok',
        'planned_steps': planned_steps,
        'steps': kept[0]['steps'],
        'samples': kept[0]['samples'],
        'epochs_used': float(len(kept)),
        **agg,
    }


def _median(xs: List[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def fit_exponent(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    """Slope of log y against log x, its 95% half-width, and R^2.

    The half-width uses a t critical value rather than 1.96: these fits have
    three or four degrees of freedom and the normal approximation would make
    an exponent of 1.4 look distinguishable from 2.
    """
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    n = len(pts)
    if n < 2:
        return float('nan'), float('nan'), float('nan')
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx <= 0:
        return float('nan'), float('nan'), float('nan')
    slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
    intercept = my - slope * mx
    resid = [p[1] - (intercept + slope * p[0]) for p in pts]
    sse = sum(r * r for r in resid)
    sst = sum((p[1] - my) ** 2 for p in pts)
    r2 = 1.0 - sse / sst if sst > 0 else float('nan')
    df = n - 2
    if df <= 0:
        return slope, float('nan'), r2
    se = math.sqrt(sse / df / sxx)
    return slope, t_crit(df) * se, r2


def group(runs: List[Dict], x_field: str) -> Dict[str, List[Dict]]:
    by_arm: Dict[str, List[Dict]] = defaultdict(list)
    for r in runs:
        arm = str(r['config'].get('align_objective', '?'))
        r['x'] = float(r['config'].get(x_field, 0) or 0)
        by_arm[arm].append(r)
    for arm in by_arm:
        by_arm[arm].sort(key=lambda r: r['x'])
    return by_arm


def check_control(runs: List[Dict]) -> List[str]:
    """Fields that must match across every point of the sweep.

    batch_size and global_batch are excluded: they are the swept axis. Anything
    else differing means the sweep is measuring two configurations rather than
    one objective at two batch sizes.
    """
    control = (
        'world_size',
        'graph_num_layers',
        'graph_hidden_dim',
        'graph_aug',
        'adversarial',
        'adv_steps',
        'max_text_length',
        'use_text',
        'use_image',
        'embed_dim',
        'sigreg_num_slices',
        'sigreg_num_quad_points',
    )
    bad = []
    for f in control:
        vals = {r['config'].get(f) for r in runs if f in r['config']}
        if len(vals) > 1:
            bad.append(f'{f}: ' + ', '.join(sorted(str(v) for v in vals)))
    return bad


def modalities(cfg: Dict[str, object]) -> int:
    return (
        1
        + int(bool(cfg.get('use_text', False)))
        + int(bool(cfg.get('use_image', False)))
    )


def report(by_arm: Dict[str, List[Dict]], args: argparse.Namespace) -> Dict[str, Dict]:
    fits: Dict[str, Dict] = {}
    xlabel = 'global B' if args.x == 'global_batch' else 'rank B'

    print(f'\n=== sweep, one epoch per point ({xlabel} on the x axis) ===')
    print(
        f'{"arm":<10}{xlabel:>10}{"steps":>7}{"smp":>5}{"epoch s":>9}'
        f'{"step ms":>9}{"align ms":>10}{"share":>8}{"ms/1k sub":>11}'
        f'{"peak GiB":>10}'
    )
    print('-' * 89)
    for arm in [a for a in ARM_ORDER if a in by_arm] + [
        a for a in by_arm if a not in ARM_ORDER
    ]:
        for r in by_arm[arm]:
            if r['status'] != 'ok':
                print(
                    f'{arm:<10}{r["x"]:>10.0f}{"":>7}{"":>5}'
                    f'{r["status"].upper():>9}'
                )
                continue
            per_1k = 1000.0 * r['step_ms'] / max(r['x'], 1.0)
            flag = '*' if r['samples'] and r['samples'] < args.min_samples else ' '
            print(
                f'{arm:<10}{r["x"]:>10.0f}{int(r["steps"]):>7}'
                f'{int(r["samples"]):>4}{flag}{r["epoch_s"]:>9.0f}'
                f'{r["step_ms"]:>9.1f}{r["loss_ms"]:>10.2f}'
                f'{100 * r["loss_fwd_share"]:>7.1f}%{per_1k:>11.2f}'
                f'{r["peak_mem_mib"] / 1024:>10.2f}'
            )
    thin = [
        r
        for rs in by_arm.values()
        for r in rs
        if r['status'] == 'ok' and r['samples'] and r['samples'] < args.min_samples
    ]
    if thin:
        print(
            f'  * fewer than {args.min_samples} sampled steps; the median is thin there.'
        )

    print(
        f'\n=== exponents: y ~ ({xlabel})^p, fit over points with x >= {args.fit_from} ==='
    )
    print(
        '  Two fits per quantity. "all" uses every point; "top" uses the larger\n'
        '  half. Both terms carry a fixed launch cost that does not scale with B,\n'
        '  and at small batches that constant is most of the measurement, which\n'
        '  pulls the fitted exponent below the true one. The same happens to peak\n'
        '  memory, where weights and optimizer state are a constant baseline. The\n'
        '  "top" column is the asymptotic claim; "all" shows how far from it the\n'
        '  small batches sit.'
    )
    print(
        f'{"arm":<10}{"quantity":<16}{"p (all)":>9}{"+/-":>8}{"R^2":>7}'
        f'{"p (top)":>10}{"+/-":>8}{"R^2":>7}   isolated'
    )
    print('-' * 90)
    for arm in [a for a in ARM_ORDER if a in by_arm] + [
        a for a in by_arm if a not in ARM_ORDER
    ]:
        pts = [
            r for r in by_arm[arm] if r['status'] == 'ok' and r['x'] >= args.fit_from
        ]
        if len(pts) < 3:
            print(f'{arm:<10}(only {len(pts)} usable point(s); a log-log fit needs 3)')
            continue
        top = pts[len(pts) // 2 :] if len(pts) >= 6 else pts[-3:]
        xs = [r['x'] for r in pts]
        xt = [r['x'] for r in top]
        fits[arm] = {}
        for label, key in (
            ('align fwd', 'loss_ms'),
            ('step', 'step_ms'),
            ('encoder fwd', 'forward_ms'),
            ('backward', 'backward_ms'),
            ('peak memory', 'peak_mem_mib'),
        ):
            p, ci, r2 = fit_exponent(xs, [r[key] for r in pts])
            pt_, cit, r2t = fit_exponent(xt, [r[key] for r in top])
            fits[arm][key] = {
                'p': p,
                'ci': ci,
                'r2': r2,
                'n': len(pts),
                'p_top': pt_,
                'ci_top': cit,
                'r2_top': r2t,
                'n_top': len(top),
            }
            ref = ''
            if key == 'loss_ms' and arm in ISOLATED:
                ip, ici = ISOLATED[arm]
                ref = f'{ip:.2f} +/- {ici:.2f}'
            print(
                f'{arm:<10}{label:<16}{p:>9.2f}{ci:>8.2f}{r2:>7.3f}'
                f'{pt_:>10.2f}{cit:>8.2f}{r2t:>7.3f}   {ref}'
            )
        print(f'{"":<10}(all: {len(pts)} points, top: {len(top)})\n')

    k = modalities(next(iter(by_arm.values()))[0]['config'])
    reach = {
        arm: max((r['x'] for r in rs if r['status'] == 'ok'), default=0.0)
        for arm, rs in by_arm.items()
    }
    print('\n=== largest batch each arm could hold ===')
    for arm, v in reach.items():
        oom = [r['x'] for r in by_arm[arm] if r['status'] == 'oom']
        tail = f'  (OOM at {min(oom):.0f})' if oom else ''
        print(f'  {arm:<10}{xlabel} = {v:.0f}{tail}')
    if len(set(reach.values())) > 1:
        hi = max(reach, key=lambda a: reach[a])
        lo = min(reach, key=lambda a: reach[a])
        print(
            f'  {hi} ran at {reach[hi] / max(reach[lo], 1):.0f}x the batch {lo} could\n'
            '  hold, on the same encoder and the same GPUs. That is the space result:\n'
            '  it is a limit on what the objective permits, not a tuning choice.'
        )

    xstar = CROSSOVER.get(k)
    if xstar and args.x == 'global_batch':
        reached = min(reach.values(), default=0.0)
        print(f'\n=== against the isolated benchmark (k={k} modalities) ===')
        print(f'  measured crossover in the isolated term: global B = {xstar}')
        print(f'  largest batch BOTH arms ran end to end:   global B = {reached:.0f}')
        if reached < xstar:
            print(
                '  The sweep stops below the crossover. Below it the volumetric term\n'
                '  is the more expensive of the two, so an end-to-end speedup is not\n'
                '  the claim to make from this figure. What it shows is the shape:\n'
                '  the alignment exponents, and that the encoder sets the step cost\n'
                '  at every batch size this model can hold.'
            )
        else:
            print(
                '  The sweep passes the crossover, so the per-subgraph step cost is\n'
                '  the number to quote: it separates where the alignment terms do.'
            )

    if by_arm:
        print('\n=== how much of the step the objective is ===')
        for arm, rs in by_arm.items():
            ok = [r for r in rs if r['status'] == 'ok']
            if not ok:
                continue
            lo, hi = ok[0], ok[-1]
            print(
                f'  {arm:<10}{100 * lo["loss_fwd_share"]:>5.1f}% at {xlabel}={lo["x"]:.0f}'
                f'  ->{100 * hi["loss_fwd_share"]:>6.1f}% at {xlabel}={hi["x"]:.0f}'
            )
        print(
            '  The rest is the encoder, tokenisation and the optimizer, which are\n'
            '  identical across arms by construction. A difference in total step time\n'
            '  larger than these shares is not attributable to the objective.'
        )
    return fits


def emit_latex(
    by_arm: Dict[str, List[Dict]],
    fits: Dict[str, Dict],
    path: str,
    args: argparse.Namespace,
) -> None:
    xlabel = 'B_{\\text{global}}' if args.x == 'global_batch' else 'B_{\\text{rank}}'
    lines = [
        '% generated by tgfm/tools/collect_batch_sweep.py',
        '\\begin{table}[t]',
        '\\caption{End-to-end cost against batch size, one epoch per point. All '
        'arms share the encoder, world size and data; they differ in the alignment '
        'objective alone. Per-step figures are medians over instrumented steps; the '
        'alignment column is its forward pass only, since its backward is fused with '
        "the encoder's. \\textsc{oom} marks a batch the arm could not hold. Epoch "
        'time is not comparable across rows: one epoch at batch $B$ has '
        '$\\lvert\\mathcal{D}\\rvert/B$ steps.}',
        '\\label{tab:batch-sweep}',
        '\\begin{center}',
        '\\begin{tabular}{llrrrrr}',
        '\\toprule',
        '\\multicolumn{1}{c}{\\bf Objective} & \\multicolumn{1}{c}{$' + xlabel + '$} & '
        '\\multicolumn{1}{c}{\\bf step (ms)} & \\multicolumn{1}{c}{\\bf align (ms)} & '
        '\\multicolumn{1}{c}{\\bf align share} & '
        '\\multicolumn{1}{c}{\\bf ms / 1k subgraphs} & '
        '\\multicolumn{1}{c}{\\bf peak (GiB)} \\\\',
        '\\midrule',
    ]
    for arm in [a for a in ARM_ORDER if a in by_arm] + [
        a for a in by_arm if a not in ARM_ORDER
    ]:
        label = ARM_LABEL.get(arm, arm).replace('_', '\\_')
        for i, r in enumerate(by_arm[arm]):
            head = label if i == 0 else ''
            if r['status'] != 'ok':
                lines.append(
                    f'{head} & {r["x"]:.0f} & \\multicolumn{{5}}{{c}}{{\\textsc{{oom}}}} \\\\'
                )
                continue
            per_1k = 1000.0 * r['step_ms'] / max(r['x'], 1.0)
            lines.append(
                f'{head} & {r["x"]:.0f} & {r["step_ms"]:.1f} & {r["loss_ms"]:.2f} & '
                f'{100 * r["loss_fwd_share"]:.1f}\\% & {per_1k:.2f} & '
                f'{r["peak_mem_mib"] / 1024:.2f} \\\\'
            )
        lines.append('\\midrule')
    lines[-1] = '\\midrule'
    for arm in [a for a in ARM_ORDER if a in fits]:
        f = fits[arm]['loss_ms']
        g = fits[arm]['step_ms']
        label = ARM_LABEL.get(arm, arm).replace('_', '\\_')
        lines.append(
            f'\\multicolumn{{7}}{{l}}{{\\footnotesize {label}: align $\\propto B^{{'
            f'{f["p_top"]:.2f} \\pm {f["ci_top"]:.2f}}}$ over the largest '
            f'{f["n_top"]} batches ($R^2={f["r2_top"]:.3f}$; '
            f'{f["p"]:.2f} over all {f["n"]}), step $\\propto B^{{'
            f'{g["p_top"]:.2f} \\pm {g["ci_top"]:.2f}}}$}} \\\\'
        )
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{center}', '\\end{table}', '']
    Path(path).write_text('\n'.join(lines))
    print(f'wrote {path}')


def emit_figure(
    by_arm: Dict[str, List[Dict]], stem: str, args: argparse.Namespace
) -> None:
    try:
        import matplotlib

        matplotlib.use('Agg')
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available; skipping the figure')
        return

    xlabel = 'global batch' if args.x == 'global_batch' else 'per-rank batch'
    panels = (
        ('loss_ms', 'alignment forward (ms)', True),
        ('per_1k', 'step cost per 1k subgraphs (ms)', False),
        ('peak_mem_mib', 'peak memory (GiB)', False),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))

    for ax, (key, ylabel, guides) in zip(axes, panels):
        any_pt = False
        for arm in [a for a in ARM_ORDER if a in by_arm]:
            pts = [r for r in by_arm[arm] if r['status'] == 'ok']
            if not pts:
                continue
            xs = [r['x'] for r in pts]
            if key == 'per_1k':
                ys = [1000.0 * r['step_ms'] / max(r['x'], 1.0) for r in pts]
            elif key == 'peak_mem_mib':
                ys = [r['peak_mem_mib'] / 1024.0 for r in pts]
            else:
                ys = [r[key] for r in pts]
            ax.plot(
                xs,
                ys,
                marker=ARM_MARKER.get(arm, 'o'),
                markersize=5,
                color=ARM_COLOR.get(arm),
                label=ARM_LABEL.get(arm, arm),
                linewidth=1.8,
            )
            any_pt = True
            # An OOM is information: mark the first batch the arm could not hold.
            oom = [r for r in by_arm[arm] if r['status'] == 'oom']
            if oom and ys:
                ax.annotate(
                    'OOM',
                    xy=(oom[0]['x'], ys[-1]),
                    xytext=(0, 9),
                    textcoords='offset points',
                    ha='center',
                    fontsize=8,
                    color=ARM_COLOR.get(arm),
                )
        ax.set_xscale('log', base=2)
        # The per-subgraph panel asks a flat-or-rising question over a narrow
        # range; a log axis turns a 20% change into a sliver and labels it
        # 5.6x10^1. Linear reads correctly there and log everywhere else.
        ax.set_yscale('linear' if key == 'per_1k' else 'log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both', alpha=0.25, linewidth=0.5)

        if guides and any_pt:
            # One reference slope per arm, anchored on that arm's own leftmost
            # point, so the eye compares the series to its predicted slope
            # rather than to a line floating somewhere else on the panel.
            xmax = max(
                r['x'] for rs in by_arm.values() for r in rs if r['status'] == 'ok'
            )
            for arm, power, style in (('volume', 1, ':'), ('infonce', 2, '--')):
                pts = [r for r in by_arm.get(arm, []) if r['status'] == 'ok']
                if not pts:
                    continue
                x0, y0 = pts[0]['x'], pts[0][key]
                ax.plot(
                    [x0, xmax],
                    [y0, y0 * (xmax / x0) ** power],
                    style,
                    color='0.45',
                    linewidth=1.0,
                    zorder=0,
                )
                ax.text(
                    xmax,
                    y0 * (xmax / x0) ** power,
                    r'  $\propto B$' if power == 1 else r'  $\propto B^2$',
                    fontsize=8,
                    color='0.35',
                    va='center',
                )

        if args.x == 'global_batch':
            k = modalities(next(iter(by_arm.values()))[0]['config'])
            xs_star = CROSSOVER.get(k)
            if xs_star:
                ax.axvline(
                    xs_star, color='0.6', linewidth=0.9, linestyle='-.', zorder=0
                )

    axes[0].legend(fontsize=8, frameon=False, loc='upper left')
    axes[0].set_title('where the objectives differ', fontsize=10)
    axes[1].set_title('what the step actually costs', fontsize=10)
    axes[2].set_title('space', fontsize=10)
    fig.suptitle(
        'End-to-end pretraining cost against batch size, one epoch per point. '
        'Dash-dot line: crossover measured on the isolated term.',
        fontsize=9,
        y=1.0,
    )
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        out = f'{stem}.{ext}'
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f'wrote {out}')
    plt.close(fig)


def main() -> None:
    args = parser.parse_args()
    runs = []
    for p in args.logs:
        r = parse_log(Path(p), args.skip_epochs)
        if r is None:
            print(f'  no TIMING lines and no fingerprint in {p}; skipped')
            continue
        runs.append(r)
    if not runs:
        print('No points parsed. Is StepTimer wired into time_main.py?')
        return

    bad = check_control(runs)
    if bad:
        print('=== control check: these differ across the sweep and should not ===')
        for b in bad:
            print(f'  {b}')
        print(
            '  batch_size is the swept axis and is expected to differ; anything\n'
            '  above means the points are not the same model.'
        )
    else:
        print('=== control check: every non-swept field matches across the sweep ===')

    by_arm = group(runs, args.x)
    fits = report(by_arm, args)

    if args.emit_latex:
        emit_latex(by_arm, fits, args.emit_latex, args)
    if args.emit_figure:
        emit_figure(by_arm, args.emit_figure, args)
    if args.out:
        Path(args.out).write_text(json.dumps({'runs': runs, 'fits': fits}, indent=2))
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
