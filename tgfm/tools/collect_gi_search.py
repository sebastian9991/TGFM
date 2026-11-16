"""Rank GramJEPA G-I search trials by best ele-fashion validation accuracy.

Reads each trial's log, takes the highest VAL-ACC over its epochs (the same
quantity mm_main selects on), and prints the trials against the control with
the sampled hyperparameters beside them.

Test accuracy is printed only from the epoch validation chose, and it is a
diagnostic here, never the ranking key.

    python tools/collect_gi_search.py
"""

import argparse
import re
from pathlib import Path

import yaml

parser = argparse.ArgumentParser(
    description='Rank GramJEPA G-I search trials.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--log-dir', type=str, default='logs/gramjepa_gi_search')
parser.add_argument('--config-dir', type=str, default='configs/gramJEPA/mm/search_gi')
parser.add_argument('--top', type=int, default=10)

SEARCHED = [
    'lambda_graph',
    'lr',
    'proj_dropout',
    'predictor_hidden_dim',
    'aug_edge_drop',
    'sigreg_num_slices',
]
EVAL_RE = re.compile(r'------------Evaluate-(\d+)------------')
PROBE_RE = re.compile(
    r'DATA: (\S+) \| METHOD: nc-probe \| VAL-ACC: ([\d.]+) \| '
    r'TEST-ACC: ([\d.]+)\+/-[\d.]+ \| TEST-F1: ([\d.]+)'
)


def read_trial(log_path: Path):
    """(best_val, epoch, test_acc, test_f1, n_epochs_seen) or None."""
    epoch, best = 0, None
    seen = 0
    for line in log_path.read_text(errors='replace').splitlines():
        m = EVAL_RE.search(line)
        if m:
            epoch = int(m.group(1))
            seen += 1
            continue
        m = PROBE_RE.search(line)
        if m:
            val, test_acc, test_f1 = (float(m.group(i)) for i in (2, 3, 4))
            if best is None or val > best[0]:
                best = (val, epoch, test_acc, test_f1)
    return None if best is None else (*best, seen)


def main() -> None:
    args = parser.parse_args()
    log_dir, cfg_dir = Path(args.log_dir), Path(args.config_dir)
    rows = []
    for log_path in sorted(log_dir.glob('trial_*.log')):
        tag = log_path.stem
        result = read_trial(log_path)
        if result is None:
            print(f'{tag}: no completed evaluation')
            continue
        cfg_path = cfg_dir / f'{tag}.yaml'
        params = {}
        if cfg_path.is_file():
            ((_, exp),) = yaml.safe_load(cfg_path.read_text())['ExperimentArguments'][
                'exp_args'
            ].items()
            params = {k: exp['model_args'][k] for k in SEARCHED}
        rows.append((tag, result, params))

    if not rows:
        print(f'No trial logs under {log_dir}')
        return

    control = next((r for r in rows if r[0] == 'trial_000'), None)
    rows.sort(key=lambda r: -r[1][0])

    head = f'{"trial":10s} {"val":>7s} {"ep":>3s} {"test":>7s} {"F1":>7s}'
    head += ' ' + ' '.join(f'{k[:12]:>12s}' for k in SEARCHED)
    print(head)
    print('-' * len(head))
    for tag, (val, ep, acc, f1, seen), params in rows[: args.top]:
        line = f'{tag:10s} {val:7.4f} {ep:3d} {acc:7.4f} {f1:7.4f}'
        line += ' ' + ' '.join(f'{params.get(k, float("nan")):12.5g}' for k in SEARCHED)
        if control is not None and tag != 'trial_000':
            line += f'  ({val - control[1][0]:+.4f} vs control)'
        print(line)

    if control is not None:
        val, ep, acc, f1, seen = control[1]
        print(f'\ncontrol (trial_000): val {val:.4f} at epoch {ep}, test {acc:.4f}')
    peaked_early = [t for t, r, _ in rows if r[1] <= 1 and r[4] >= 3]
    if peaked_early:
        print(
            f'{len(peaked_early)} trial(s) peaked at epoch 1 with >=3 epochs run: '
            'validation degrades with training, so look at lambda_graph and lr '
            'in those rows before trusting the ranking.'
        )


if __name__ == '__main__':
    main()
