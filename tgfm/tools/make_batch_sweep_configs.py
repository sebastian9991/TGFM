"""Emit an end-to-end batch-size sweep: one config per (arm, batch size).

tgfm/tools/bench_alignment_cost.py measures the alignment term on synthetic
embeddings and shows the separation cleanly: cost grows like B for the
volumetric term and like B^2 for InfoNCE, with a measured crossover at global
B ~ 2554 for graph+text (k=2). That benchmark answers "is the term linear or
quadratic". It does not answer "does it matter in a training step", because a
training step also runs the encoder, whose cost is linear in B and, at the
batch sizes we can fit, much larger than either alignment term.

This sweep answers the second question with the real loop. It varies the batch
size and runs one epoch per point per arm, holding everything else fixed, so
the arms differ in align_objective alone.

Read the result on three quantities, not on epoch_s. One epoch at batch B has
pool/(B*W) steps, so epoch_s falls with B for reasons that have nothing to do
with the objective and is not comparable across points.

    loss_ms      alignment forward, isolated. This is where the B vs B^2
                 separation lives, and it is the number to fit.
    step_ms/B    per-subgraph step cost. Flat means the objective is free at
                 this scale; rising means it is not. This is the honest
                 end-to-end claim.
    peak_mem_mib space. Both arms carry the encoder's activations, which grow
                 like B*N_max*C because GPSConv pads every subgraph to N_max,
                 so both rise; InfoNCE additionally holds a B_global^2 logit
                 matrix, which is what separates them.

Sizing. batch_size in the config is PER RANK; the quadratic term is in the
GLOBAL batch, because InfoNCE all-gathers before forming logits. The default
grid reaches a global batch of 4096 at world size 4, which is past the k=2
crossover. Expect the largest points to OOM on the encoder rather than on the
loss --- report those as OOM rather than dropping them; an objective that
cannot reach a batch size is a result about that objective.

    python tgfm/tools/make_batch_sweep_configs.py --base configs/legtjepa/volume.yaml \\
        --arms volume infonce --world-size 4
    python tgfm/tools/make_batch_sweep_configs.py --base configs/legtjepa/volume.yaml \\
        --arms volume infonce mse --batch-sizes 64 128 256 512 --world-size 4
"""

import argparse
import copy
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

parser = argparse.ArgumentParser(
    description='Emit matched (arm x batch size) configs for an end-to-end sweep.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--base', type=str, required=True, help='Existing config to derive from.'
)
parser.add_argument(
    '--arms',
    type=str,
    nargs='+',
    default=['volume', 'infonce'],
    help='align_objective values. time_main.py maps volume->GramJEPA, '
    'infonce->GraphCLIP, mse->LeGTJEPA.',
)
parser.add_argument(
    '--batch-sizes',
    type=int,
    nargs='+',
    default=[64, 128, 256, 384, 512, 768, 1024],
    help='PER-RANK batch sizes. Spread for the fit, and extra points at the top '
    'because that is where the asymptote is and where the OOMs start.',
)
parser.add_argument(
    '--world-size',
    type=int,
    default=4,
    help='Ranks the sweep will run on. Used for the global batch in the plan '
    'and for the launch script; it is not written into the config.',
)
parser.add_argument('--out-dir', type=str, default='configs/legtjepa/batch_sweep')
parser.add_argument('--tag', type=str, default='bsweep')
parser.add_argument(
    '--entry',
    type=str,
    default='tgfm/experiments/leGTjepa/time_main.py',
    help='Training script the launch lines call.',
)
parser.add_argument(
    '--timing-samples',
    type=int,
    default=24,
    help='Sampled steps per epoch, fixed across batch sizes by StepTimer.plan.',
)
parser.add_argument(
    '--launch',
    type=str,
    default='',
    help='Path for the sequential launch script. Default depends on --launch-style.',
)
parser.add_argument(
    '--launch-style',
    type=str,
    default='sbatch',
    choices=('sbatch', 'interactive'),
    help="'sbatch' writes a batch script with its own SLURM header. "
    "'interactive' writes a plain script to run inside salloc: no srun, and "
    'it skips points whose log already has a TIMING line so it can be re-run '
    'after a disconnect.',
)

# Must be identical across every emitted config, or the sweep measures the
# encoder's configuration drifting rather than the objective.
CONTROLLED = (
    'source_data',
    'max_text_length',
    'graph_in_dim',
    'graph_hidden_dim',
    'graph_num_layers',
    'graph_pe_dim',
    'attn_type',
    'attn_dropout',
    'embed_dim',
    'proj_hidden_dim',
    'proj_dropout',
    'predictor_depth',
    'predictor_hidden_dim',
    'predictor_dropout',
    'sigreg_num_slices',
    'sigreg_num_quad_points',
    'sigreg_t_max',
    'lambda_graph',
    'lambda_text',
    'lambda_image',
    'graph_aug',
    'aug_feat_drop',
    'aug_edge_drop',
    'adversarial',
    'adv_step_size',
    'adv_steps',
    'lr',
    'weight_decay',
    'warmup_steps',
    'lr_schedule',
    'text_input_mode',
    'text_in_dim',
    'image_in_dim',
    'mm_feat_name',
    'use_text',
    'use_image',
    'graph_feat',
    'freeze_text_projection',
    'freeze_image_projection',
    'freeze_text_backbone',
    'text_model_id',
)
# Deliberately NOT controlled: batch_size is the swept axis.
SWEPT = 'batch_size'


def emit_launch(
    path: Path, jobs: List[Tuple[str, int, Path]], args: argparse.Namespace
) -> None:
    """Every point in sequence inside one allocation.

    Splitting the sweep across allocations would let SLURM place points on
    different nodes, and a timing curve assembled from two GPU models is not a
    curve. Each point is its own torchrun, so an OOM at the top of the grid
    ends that point and not the sweep.

    Two styles. `sbatch` writes its own SLURM header and submits as a batch
    job. `interactive` assumes you are already inside salloc on the compute
    node: no header, no srun (torchrun launches the ranks itself, and srun
    inside an interactive allocation competes with it for the step), and each
    point is skipped when its log already carries a TIMING line, so the script
    can be re-run after a disconnect without repeating finished work.
    """
    inter = args.launch_style == 'interactive'
    lines = ['#!/bin/bash']
    if not inter:
        lines += [
            '#SBATCH --job-name=bsweep',
            f'#SBATCH --gpus-per-node={args.world_size}',
            '#SBATCH --cpus-per-task=24',
            '#SBATCH --mem=128G',
            '#SBATCH --time=12:00:00',
            '#SBATCH --output=logs/bsweep_%j.out',
        ]
    lines += [
        '',
        '# Every point runs in this one allocation, so the whole curve comes',
        '# from the same GPUs. An OOM ends its point and the sweep continues.',
        'set -u',
        '',
        f'export TGFM_TIMING_SAMPLES={args.timing_samples}',
    ]
    if inter:
        lines += [
            '',
            '# Re-running skips points that already produced a TIMING line.',
            '# FORCE=1 ./' + path.name + ' re-runs everything.',
            'FORCE=${FORCE:-0}',
            'ONLY=${ONLY:-}          # e.g. ONLY=infonce, or ONLY=b512',
        ]
    lines.append('')

    runner = '' if inter else 'srun --gres-flags=allow-task-sharing '
    for arm, bs, cfg in jobs:
        stem = cfg.stem
        head = f'=== {arm} batch_size={bs} (global {bs * args.world_size}) ==='
        if inter:
            lines += [
                f'if [[ -n "$ONLY" && "{stem}" != *"$ONLY"* ]]; then',
                f'  echo "skip (ONLY=$ONLY): {stem}"',
                f'elif [[ "$FORCE" != 1 ]] && grep -q "TIMING epoch" logs/{stem}.log '
                '2>/dev/null; then',
                f'  echo "skip (done): {stem}"',
                'else',
                f'  echo "{head}"',
                f'  uv run --group graphclip \\',
                f'    torchrun --standalone --nproc_per_node={args.world_size} \\',
                f'    {args.entry} --config-file {cfg} \\',
                f'    || echo "POINT-FAILED arm={arm} batch_size={bs}"',
                'fi',
                '',
            ]
        else:
            lines += [
                f'echo "{head}"',
                f'{runner}uv run --group graphclip \\',
                f'    torchrun --standalone --nproc_per_node={args.world_size} \\',
                f'    {args.entry} --config-file {cfg} \\',
                f'    || echo "POINT-FAILED arm={arm} batch_size={bs}"',
                '',
            ]

    logs = ' '.join(f'logs/{c.stem}.log' for _, _, c in jobs)
    lines += [
        'echo "=== collect ==="',
        f'python tgfm/tools/collect_batch_sweep.py {logs} \\',
        '    --emit-latex analysis/batch_sweep.tex \\',
        '    --emit-figure analysis/batch_sweep',
        '',
    ]
    path.write_text('\n'.join(lines))
    path.chmod(0o755)


def main() -> None:
    args = parser.parse_args()
    base_path = Path(args.base)
    if not base_path.is_file():
        raise SystemExit(f'{base_path} not found (run from the repo root)')
    base = yaml.safe_load(base_path.read_text())

    exp_args = base['ExperimentArguments']['exp_args']
    if len(exp_args) != 1:
        raise SystemExit(
            f'{base_path} defines {len(exp_args)} experiments; this needs exactly one.'
        )
    (base_exp_name,) = exp_args
    base_exp = exp_args[base_exp_name]
    controls: Dict[str, object] = {
        k: base_exp['model_args'][k] for k in CONTROLLED if k in base_exp['model_args']
    }

    batch_sizes = sorted(set(args.batch_sizes))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs: List[Tuple[str, int, Path]] = []

    # Ascending: a partial sweep that OOMs at the top is still fittable,
    # whereas one that dies at the bottom leaves nothing.
    for bs in batch_sizes:
        for arm in args.arms:
            cfg = copy.deepcopy(base)
            exp = copy.deepcopy(base_exp)
            name = f'{args.tag}_{arm}_b{bs}'
            cfg['ExperimentArguments']['exp_args'] = {name: exp}
            cfg['MetaArguments']['log_file_path'] = f'{name}.log'
            # verbose gates wandb AND the end-of-run zero-shot evaluation,
            # which time_main runs on the final epoch regardless of
            # eval_every_epochs. With epochs=1 every point is a final epoch, so
            # leaving it on would put an hour of evaluation after each of them.
            cfg['MetaArguments']['verbose'] = False
            if 'wandb_run_name' in cfg['MetaArguments']:
                cfg['MetaArguments']['wandb_run_name'] = name

            exp['model_args'].update(controls)
            exp['model_args']['align_objective'] = arm
            exp['model_args'][SWEPT] = bs
            exp['model_args']['epochs'] = 1
            # scalar_logs calls .item() on every logged loss, which is a host
            # sync. Its frequency is per step, so leaving it on would add a
            # cost that scales with the step count and therefore varies with B.
            exp['model_args']['log_every_steps'] = 10**9
            exp['data_args']['eval_every_epochs'] = 10**9

            path = out_dir / f'{name}.yaml'
            path.write_text(
                yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
            )
            jobs.append((arm, bs, path))

    launch = Path(
        args.launch
        or (
            'run_batch_sweep.sh'
            if args.launch_style == 'interactive'
            else 'launch_batch_sweep.sh'
        )
    )
    emit_launch(launch, jobs, args)

    print(f'base: {base_path}  (experiment {base_exp_name!r})')
    print(f'swept: batch_size (per rank) over {batch_sizes}')
    print(f'arms:  {", ".join(args.arms)}')
    print(f'pinned across every config: {len(controls)} fields')
    for k in sorted(controls):
        print(f'    {k} = {controls[k]}')
    print(f'\n{len(jobs)} configs in {out_dir}/')
    print(f'\n{"per-rank B":>11}{"global B":>10}   note')
    for bs in batch_sizes:
        g = bs * args.world_size
        note = 'past the measured crossover (2554)' if g >= 2554 else ''
        print(f'{bs:>11}{g:>10}   {note}')
    print(
        '\nPer-rank batch sets the encoder cost; global batch sets the InfoNCE\n'
        'logit matrix. GPSConv pads every subgraph to N_max, so the top of the\n'
        'grid will likely OOM on encoder activations before the loss. That is a\n'
        'result, not a failure: collect_batch_sweep reports it as OOM.'
    )
    if args.launch_style == 'interactive':
        print(
            f'\n  salloc --gres=gpu:{args.world_size} --cpus-per-task=24 '
            '--mem=128G --time=12:00:00\n'
            f'  ./{launch}\n\n'
            '  Run it under tmux or screen: an interactive allocation dies with\n'
            '  the login session. Re-running skips finished points; ONLY=infonce\n'
            '  or ONLY=b512 restricts it, FORCE=1 redoes everything.'
        )
    else:
        print(f'\n  sbatch {launch}')


if __name__ == '__main__':
    main()
