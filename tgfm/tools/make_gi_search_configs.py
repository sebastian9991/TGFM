"""Generate random-search trial configs for GramJEPA G-I.

Scored on ele-fashion validation accuracy only. Dropping books-nc from
data_args.target_data removes the ~50 min per-eval ego-subgraph build, which is
what makes a search affordable; the winner is then re-run on both targets with
the full epoch budget.

Trial 000 is the current tuned config unchanged, so every other trial is read
against a control that ran under the same code and the same search budget.

Search space, ordered by expected effect on the probe:
    lambda_graph          SIGReg weight on the graph marginal. The G-I run
                          peaked at epoch 2 of 10 and fell after, which is what
                          an under-regularized encoder collapsing onto the
                          alignment target looks like.
    lr                    2.02e-4 was swept on a different setup (graph+text,
                          384-d, trainable target).
    proj_dropout          the probe reads graph_projection's output, so dropout
                          here directly removes linearly decodable information.
    predictor_hidden_dim  a 2048-wide predictor can satisfy the alignment on its
                          own and leave the encoder underconstrained.
    aug_edge_drop         0.342 removes a third of the edges from subgraphs
                          already thinned by fanout 12.
    sigreg_num_slices     128 random directions is thin coverage of a 768-d
                          sphere for the Epps-Pulley statistic.

Everything else is held at the tuned values, including graph_feat='image',
freeze_image_projection=true and embed_dim=768.

    python tools/make_gi_search_configs.py --trials 24
    sbatch launches/launch_gramjepa_gi_search.sh
    python tools/collect_gi_search.py
"""

import argparse
import copy
import math
from pathlib import Path

import numpy as np
import yaml

parser = argparse.ArgumentParser(
    description='Write random-search trial configs for GramJEPA G-I.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--base',
    type=str,
    default='configs/gramJEPA/mm/gramjepa_gi_tuned.yaml',
    help='Config the trials are derived from. Trial 000 is this, unchanged '
    'except for the search-harness fields.',
)
parser.add_argument('--out-dir', type=str, default='configs/gramJEPA/mm/search_gi')
parser.add_argument('--trials', type=int, default=24)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument(
    '--target', type=str, default='ele-fashion', help='Scored target only.'
)
parser.add_argument(
    '--eval-seeds',
    type=int,
    nargs='+',
    default=[0, 1],
    help='Probe seeds per eval. Accuracy std across seeds was ~1e-3, so 2 is '
    'enough to rank trials; the final run uses all 5.',
)


def sample(rng: np.random.Generator) -> dict:
    """One draw from the search space."""

    def loguniform(low: float, high: float) -> float:
        return float(math.exp(rng.uniform(math.log(low), math.log(high))))

    return {
        'lambda_graph': loguniform(5.0e-4, 1.5e-2),
        'lr': loguniform(6.0e-5, 4.0e-4),
        'proj_dropout': float(rng.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])),
        'predictor_hidden_dim': int(rng.choice([512, 1024, 2048])),
        'aug_edge_drop': float(rng.choice([0.0, 0.10, 0.20, 0.34])),
        'sigreg_num_slices': int(rng.choice([128, 256, 512])),
    }


def main() -> None:
    args = parser.parse_args()
    base_path = Path(args.base)
    if not base_path.is_file():
        raise FileNotFoundError(f'{base_path} not found (run from the repo root)')
    base = yaml.safe_load(base_path.read_text())
    (base_exp,) = base['ExperimentArguments']['exp_args']

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob('trial_*.yaml'):
        stale.unlink()

    rng = np.random.default_rng(args.seed)
    rows = []
    for t in range(args.trials):
        cfg = copy.deepcopy(base)
        exp_name = f'GramJEPA_GI_search_t{t:03d}'
        exp = cfg['ExperimentArguments']['exp_args'].pop(base_exp)
        cfg['ExperimentArguments']['exp_args'][exp_name] = exp
        model_args, data_args = exp['model_args'], exp['data_args']

        # Harness settings, identical across trials including the control.
        cfg['MetaArguments']['log_file_path'] = (
            f'logs/gramjepa_gi_search/trial_{t:03d}.log'
        )
        model_args['epochs'] = args.epochs
        data_args['target_data'] = args.target
        data_args['eval_seeds'] = list(args.eval_seeds)
        data_args['eval_every_epochs'] = 1  # the peak came early; watch for it

        overrides = {} if t == 0 else sample(rng)
        model_args.update(overrides)

        (out_dir / f'trial_{t:03d}.yaml').write_text(
            yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
        )
        rows.append((t, exp_name, overrides))

    keys = sorted({k for _, _, o in rows for k in o})
    print(f'{len(rows)} configs -> {out_dir}')
    print('trial  ' + '  '.join(f'{k:>20s}' for k in keys))
    for t, _, o in rows:
        cells = []
        for k in keys:
            v = o.get(k)
            cells.append('  (control)'.rjust(20) if v is None else f'{v:>20.6g}')
        print(f'{t:5d}  ' + '  '.join(cells))
    print(f'\nsbatch --array=0-{len(rows) - 1}%4 launches/launch_gramjepa_gi_search.sh')


if __name__ == '__main__':
    main()
