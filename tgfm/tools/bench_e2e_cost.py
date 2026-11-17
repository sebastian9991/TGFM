r"""End-to-end per-step cost of the alignment objectives, encoders included.

The end-to-end counterpart of tools/bench_alignment_cost.py. That tool times
the alignment term alone on synthetic embeddings; this one times a full
training step (encoder forward, objective, backward, AdamW step) of the real
MM-Graph models built from the real YAML configs, and answers:

  * At our actual 2- and 3-modality setups and batch sizes, what are the step
    time and peak memory of LeGramJEPA (volume + SIGReg) against GraphCLIP
    (InfoNCE) and pairwise L_cross + SIGReg?
  * How does that cost split into a part that grows with k and a part that
    grows with B?

Arms (``--arms``). All LeGTJEPA arms share one encoder, projections, data and
optimizer; only the objective differs:

    volume        LeGTJEPAVolumeLoss        Gramian volume + SIGReg (LeGramJEPA)
    pairwise      LeGTJEPALoss              pairwise L_cross + SIGReg, k(k-1) terms
    infonce       MultiModalInfoNCELoss     symmetric InfoNCE over all C(k,2) pairs
    encoder_only  trivial O(B d) loss       the encoder baseline. Subtracting it
                                            from an arm isolates that objective's
                                            forward AND backward end to end.
    graphclip     GraphCLIPMM + GraphCLIPLoss  the GraphCLIP model as published
                                            (GPS head, raw partner features, no
                                            projection). k = 2 only; needs the
                                            vendored repo (GRAPHCLIP_ROOT).

Separating k from B. Run every arm at k = 2 (G-T config) and k = 3 (G-T-I
config) over the same sweep of global batch sizes. The summary then reports,
per arm, (i) the log-log slope of step time against B at fixed k -- the B
exponent -- and (ii) the ratio t(k=3)/t(k=2) at fixed B -- the k cost. The
encoder_only arm gives the same two numbers for the encoder, so the objective's
own increments (arm minus encoder_only) can be read off directly.

Batch sizes are GLOBAL. Under torchrun with W ranks each rank gets B/W, the
models are wrapped in DDP with SyncBatchNorm as in mm_main.py, and InfoNCE /
SIGReg all-gather exactly as in training. Single-process runs use one GPU at
the full global B, which is the right per-rank cost for InfoNCE (every rank
builds the global B x B logits anyway) but overstates the encoder's per-rank
cost by W; run under torchrun for the training-faithful numbers.

Data. ``--data synthetic`` builds random ego-subgraphs with the real feature
widths, PE width and subgraph-size range, so it runs anywhere (laptop smoke
tests). ``--data real`` samples batches from the MM-Graph pretraining pool via
mm_main.load_source_graphs and prints their node/edge statistics, so the
synthetic sizes can be matched to them. Either way a fixed set of batches is
built once and kept in pinned host memory; the host-to-device copy is timed as
``data_ms`` and is NOT part of ``step_ms``.

Timing. Per configuration: a fresh model and optimizer, ``--warmup`` untimed
steps, then ``--iters`` steps each bracketed by CUDA events at the forward /
loss / backward / optim boundaries and synchronised at the end of the step.
Peak memory is max_memory_allocated over the timed steps (params, grads, Adam
state and activations), reset after warmup. Medians and the 25-75% range are
reported; ``--seeds`` repeats the whole configuration. Graph augmentation is
off in every arm so the arms differ in the objective alone. An OOM is recorded
as such and the sweep continues.

    # laptop smoke test
    python tgfm/tools/bench_e2e_cost.py --data synthetic --batches 32 64 128

    # cluster, 4 GPUs, training-faithful
    torchrun --standalone --nproc_per_node=4 tgfm/tools/bench_e2e_cost.py \
        --data real --batches 1800 3600 7200 --out bench_e2e.json
"""

import argparse
import copy
import gc
import json
import logging
import math
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import yaml
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import Batch, Data

from tgfm.models.legtjepa import LeGTJEPA
from tgfm.models.losses.infonce_mm import MultiModalInfoNCELoss
from tgfm.models.losses.legtjepaloss import LeGTJEPALoss
from tgfm.models.losses.volumeloss import LeGTJEPAVolumeLoss
from tgfm.utils.args import LeGTJEPAArguments, get_model_args
from tgfm.utils.mm_features import center_text_feature
from tgfm.utils.path import get_root_dir

ARMS = ('volume', 'pairwise', 'infonce', 'encoder_only', 'graphclip')
OBJECTIVE_OF = {
    'volume': 'volume',
    'pairwise': 'mse',
    'infonce': 'infonce',
    'encoder_only': 'volume',
    'graphclip': 'infonce',
}
REGIONS = ('data', 'forward', 'loss', 'backward', 'optim')

parser = argparse.ArgumentParser(
    description='End-to-end per-step cost of the alignment objectives.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--k2-config',
    default='configs/gramJEPA/mm/gramjepa_gt.yaml',
    help='G-T config (k = 2). 12-layer GPS, 1024 wide: GraphCLIP Base.',
)
parser.add_argument(
    '--k3-config',
    default='configs/gramJEPA/mm/gramjepa_gti.yaml',
    help='G-T-I config (k = 3), same encoder as --k2-config.',
)
parser.add_argument(
    '--graphclip-config', default='configs/graphCLIP_mm/mm/graphclip_gt.yaml'
)
parser.add_argument(
    '--arms',
    nargs='+',
    choices=ARMS,
    default=['volume', 'pairwise', 'infonce', 'encoder_only'],
)
parser.add_argument('--k', type=int, nargs='+', choices=[2, 3], default=[2, 3])
parser.add_argument(
    '--batches',
    type=int,
    nargs='+',
    default=[900, 1800, 3600, 7200],
    help='GLOBAL batch sizes (split across ranks under torchrun).',
)
parser.add_argument('--warmup', type=int, default=20)
parser.add_argument('--iters', type=int, default=100)
parser.add_argument('--seeds', type=int, nargs='+', default=[0])
parser.add_argument('--data', choices=['synthetic', 'real'], default='synthetic')
parser.add_argument(
    '--n-cached-batches',
    type=int,
    default=4,
    help='Distinct batches built once and cycled.',
)
parser.add_argument(
    '--syn-nodes',
    type=int,
    nargs=2,
    default=[20, 157],
    help='Synthetic subgraph size range [min, max]. 157 = 1 + 12 + 12^2, '
    'the cap of the 2-hop fanout-12 sampler. Match to the real stats.',
)
parser.add_argument(
    '--syn-degree',
    type=float,
    default=6.0,
    help='Synthetic mean out-degree within a subgraph.',
)
parser.add_argument(
    '--fixed-graph-layers',
    type=int,
    default=0,
    help='Override graph_num_layers (0 keeps the config). Debug only.',
)
parser.add_argument(
    '--checkpoint-encoder',
    action='store_true',
    help='Activation checkpointing on every graph conv layer. Same math, about '
    'one extra encoder forward per step, several times less activation memory: '
    'lets the per-GPU batch, and so the global B, grow on the same hardware. '
    'Applied identically to every arm.',
)
parser.add_argument(
    '--emulate-world',
    type=int,
    default=1,
    help='Single process only. Emulate the per-GPU cost of a W-GPU DDP job: the '
    'encoder runs on B/W samples, and every cross-rank gather in the losses '
    '(InfoNCE logits, SIGReg) is replaced by tiling the local embeddings W times, '
    'so the loss sees the full global B exactly as each rank would. Leaves out '
    'communication time and SyncBatchNorm. Use for global B beyond what the real '
    'GPUs can hold; an OOM is caught cleanly, unlike under torchrun.',
)
parser.add_argument('--out', default='', help='Write raw results as JSON (rank 0).')
parser.add_argument(
    '--emit-markdown', default='', help='Write the summary table (rank 0).'
)


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #


def setup() -> Tuple[int, int, torch.device]:
    if 'LOCAL_RANK' in os.environ:
        dist.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        return (
            dist.get_rank(),
            dist.get_world_size(),
            torch.device(f'cuda:{local_rank}'),
        )
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required: timings are CUDA events.')
    return 0, 1, torch.device('cuda:0')


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def load_yaml(path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(MetaArguments dict, first experiment's model_args dict)."""
    cfg = yaml.safe_load((get_root_dir() / path).read_text())
    exp = next(iter(cfg['ExperimentArguments']['exp_args'].values()))
    return cfg['MetaArguments'], dict(exp['model_args'])


def build_args(
    cfg: Dict[str, Any], arm: str, per_rank_batch: int, layers: int
) -> LeGTJEPAArguments:
    cfg = copy.deepcopy(cfg)
    cfg['align_objective'] = OBJECTIVE_OF[arm]
    cfg['batch_size'] = per_rank_batch
    cfg['graph_aug'] = False  # identical across arms: objective is the only change
    cfg['adversarial'] = False
    if layers:
        cfg['graph_num_layers'] = layers
    if arm == 'graphclip':
        cfg['model'] = 'GraphCLIP'
    args = get_model_args(cfg)
    assert isinstance(args, LeGTJEPAArguments)
    return args


class EncoderOnlyLoss(torch.nn.Module):
    """O(B d) stand-in that sends gradient through every projection.

    Gives the cost of encoder forward + backward + optimizer with an objective
    of negligible cost, so arm - encoder_only is the objective's own share.
    """

    def forward(self, out: Dict[str, Tensor]) -> Dict[str, Tensor]:
        loss = sum(out[k].pow(2).mean() for k in ('z_g', 'z_t', 'z_image') if k in out)
        return {'loss': loss}


class _Checkpointed(torch.nn.Module):
    """Recompute the wrapped layer's activations in backward instead of storing them."""

    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        from torch.utils.checkpoint import checkpoint

        # Non-reentrant: works under DDP and restores the RNG state, so dropout
        # masks match between the forward and the recompute.
        return checkpoint(self.layer, *args, use_reentrant=False, **kwargs)


def checkpoint_graph_convs(model: torch.nn.Module) -> int:
    """Wrap every layer of every ``convs`` ModuleList (LeGTJEPA and GraphCLIP GPS)."""
    n = 0
    for module in list(model.modules()):
        convs = getattr(module, 'convs', None)
        if isinstance(convs, torch.nn.ModuleList):
            for i, layer in enumerate(convs):
                if not isinstance(layer, _Checkpointed):
                    convs[i] = _Checkpointed(layer)
                    n += 1
    if n == 0:
        raise RuntimeError('--checkpoint-encoder: no graph conv layers found to wrap')
    return n


def emulate_gathers(world: int) -> None:
    """Make every loss module's gather_embeddings tile the local batch W times.

    The tiled copies keep autograd, so the backward through the B x B logits
    (InfoNCE) and the B-sample SIGReg statistics is the per-rank work of a real
    W-rank job. Values are not meaningful (rows repeat); costs are.
    """
    import tgfm.models.losses.infonce_mm as m_infonce
    import tgfm.models.losses.legtjepaloss as m_pair
    import tgfm.models.losses.volumeloss as m_vol

    def tiled(z: Tensor) -> Tensor:
        return z.repeat(world, 1)

    mods = [m_infonce, m_pair, m_vol]
    try:
        import tgfm.models.losses.graphclip_loss as m_gc

        mods.append(m_gc)
    except Exception:  # vendored GraphCLIP absent; its arm is skipped anyway
        pass
    for mod in mods:
        mod.gather_embeddings = tiled


def build_model(
    args: LeGTJEPAArguments,
    arm: str,
    device: torch.device,
    world_size: int,
    checkpoint_encoder: bool = False,
) -> Tuple[torch.nn.Module, torch.nn.Module, torch.optim.Optimizer]:
    if arm == 'graphclip':
        from tgfm.evaluation.graphclip_mm_adapter import GraphCLIPMM
        from tgfm.models.losses.graphclip_loss import GraphCLIPLoss

        model: torch.nn.Module = GraphCLIPMM(args)
        criterion: torch.nn.Module = GraphCLIPLoss()
    else:
        model = LeGTJEPA(args)
        # Same freezing as mm_main.py: frozen partner projections drop their
        # predictor; predictors are unused by infonce and by the baseline.
        frozen = []
        if args.use_text and args.freeze_text_projection:
            frozen.append('text_predictor')
        if args.use_image and args.freeze_image_projection:
            frozen.append('image_predictor')
        if arm in ('infonce', 'encoder_only'):
            frozen += ['graph_predictor', 'text_predictor', 'image_predictor']
        for name in frozen:
            if hasattr(model, name):
                for p in getattr(model, name).parameters():
                    p.requires_grad = False
        criterion = {
            'volume': LeGTJEPAVolumeLoss,
            'pairwise': LeGTJEPALoss,
            'infonce': MultiModalInfoNCELoss,
        }.get(arm, lambda _a: EncoderOnlyLoss())(args)

    if checkpoint_encoder:
        checkpoint_graph_convs(model)
    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    criterion = criterion.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    params += [p for p in criterion.parameters() if p.requires_grad]
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    return model, criterion, optimizer


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


def synthetic_batches(
    args: LeGTJEPAArguments,
    n_batches: int,
    per_rank: int,
    node_range: Tuple[int, int],
    degree: float,
    seed: int,
) -> List[Batch]:
    """Random ego-subgraphs with the real widths: x, 32-d RWPE, center index, image_x."""
    g = torch.Generator().manual_seed(seed)
    lo, hi = node_range
    out = []
    for _ in range(n_batches):
        items = []
        for _ in range(per_rank):
            n = int(torch.randint(lo, hi + 1, (1,), generator=g))
            m = max(1, int(n * degree))
            ei = torch.randint(0, n, (2, m), generator=g)
            items.append(
                Data(
                    x=torch.randn(n, args.graph_in_dim, generator=g),
                    pe=torch.rand(n, 32, generator=g),
                    edge_index=ei,
                    root_n_index=0,
                    image_x=torch.randn(1, args.image_in_dim, generator=g),
                    num_nodes=n,
                )
            )
        out.append(Batch.from_data_list(items).pin_memory())
    return out


_POOL: Dict[Tuple[str, str, str], Any] = {}


def real_batches(
    args: LeGTJEPAArguments, n_batches: int, per_rank: int, seed: int, rank: int
) -> List[Batch]:
    from torch_geometric.loader import DataLoader

    from tgfm.experiments.leGTjepa.mm_main import load_source_graphs

    # The pretraining pool depends only on these fields; load it once.
    key = (args.source_data, args.graph_feat, args.mm_feat_name)
    if key not in _POOL:
        _POOL[key] = load_source_graphs(None, args)  # meta_args is unused there
    graphs = _POOL[key]
    g = torch.Generator().manual_seed(seed * 1000 + rank)
    loader = DataLoader(
        graphs,
        batch_size=per_rank,
        shuffle=True,
        generator=g,
        num_workers=8,
        drop_last=True,
    )
    out = []
    for i, b in enumerate(loader):
        if i >= n_batches:
            break
        out.append(b.pin_memory())
    return out


def batch_stats(batches: List[Batch]) -> Dict[str, float]:
    nodes = [b.num_nodes / b.num_graphs for b in batches]
    edges = [b.edge_index.size(1) / b.num_graphs for b in batches]
    return {
        'nodes_per_subgraph': statistics.fmean(nodes),
        'edges_per_subgraph': statistics.fmean(edges),
    }


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #


def run_step(
    model,
    criterion,
    optimizer,
    args,
    batch_cpu,
    device,
    events: Optional[List[torch.cuda.Event]],
) -> None:
    def mark(i: int) -> None:
        if events is not None:
            events[i].record()

    mark(0)
    optimizer.zero_grad(set_to_none=True)
    batch = batch_cpu.to(device, non_blocking=True)
    text_x = center_text_feature(batch, args) if args.use_text else None
    image_x = batch.image_x if args.use_image else None
    mark(1)
    out = model(batch, text_x, image_x=image_x)
    mark(2)
    losses = criterion(out)
    mark(3)
    losses['loss'].backward()
    mark(4)
    optimizer.step()
    mark(5)


def q(vals: List[float], p: float) -> float:
    s = sorted(vals)
    i = (len(s) - 1) * p
    lo, hi = math.floor(i), math.ceil(i)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def reduce_max(x: List[float], device: torch.device) -> List[float]:
    if not is_dist():
        return x
    t = torch.tensor(x, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.tolist()


def time_config(
    model, criterion, optimizer, args, batches, device, warmup: int, iters: int
) -> Dict[str, float]:
    model.train()
    for i in range(warmup):
        run_step(
            model, criterion, optimizer, args, batches[i % len(batches)], device, None
        )
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    per_region: Dict[str, List[float]] = {r: [] for r in REGIONS + ('step',)}
    for i in range(iters):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
        run_step(
            model, criterion, optimizer, args, batches[i % len(batches)], device, ev
        )
        torch.cuda.synchronize(device)
        times = [ev[j].elapsed_time(ev[j + 1]) for j in range(5)]
        # Under DDP the slowest rank gates the step: take the per-step max.
        times = reduce_max(times, device)
        for r, t in zip(REGIONS, times):
            per_region[r].append(t)
        per_region['step'].append(sum(times[1:]))  # step excludes data
    peak = reduce_max([torch.cuda.max_memory_allocated(device) / 2**20], device)[0]

    res: Dict[str, float] = {'peak_mem_mib': peak}
    for r, vals in per_region.items():
        res[f'{r}_ms'] = statistics.median(vals)
        res[f'{r}_ms_p25'] = q(vals, 0.25)
        res[f'{r}_ms_p75'] = q(vals, 0.75)
    return res


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #


def slope(xs: List[float], ys: List[float]) -> float:
    lx, ly = [math.log(x) for x in xs], [math.log(y) for y in ys]
    mx, my = statistics.fmean(lx), statistics.fmean(ly)
    den = sum((a - mx) ** 2 for a in lx)
    return (
        sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / den if den else float('nan')
    )


def summarize(results: List[Dict[str, Any]]) -> str:
    ok = [r for r in results if r['status'] == 'ok']
    # Median over seeds per (arm, k, B).
    agg: Dict[Tuple[str, int, int], Dict[str, float]] = {}
    for key in {(r['arm'], r['k'], r['global_batch']) for r in ok}:
        rs = [r for r in ok if (r['arm'], r['k'], r['global_batch']) == key]
        agg[key] = {
            m: statistics.median(r[m] for r in rs)
            for m in ('step_ms', 'loss_ms', 'peak_mem_mib')
        }
    lines = [
        '## Per-step cost (median over timed steps and seeds)',
        '',
        '| arm | k | global B | step ms | loss fwd ms | step − encoder ms | peak MiB |',
        '|---|---|---|---|---|---|---|',
    ]
    for arm, k, b in sorted(agg):
        a = agg[(arm, k, b)]
        base = agg.get(('encoder_only', k, b))
        extra = (
            f"{a['step_ms'] - base['step_ms']:.1f}"
            if base and arm not in ('encoder_only', 'graphclip')  # graphclip: different model
            else '–'
        )
        lines.append(
            f"| {arm} | {k} | {b} | {a['step_ms']:.1f} | {a['loss_ms']:.2f} | "
            f"{extra} | {a['peak_mem_mib']:.0f} |"
        )
    for arm, k, b in sorted(
        {(r['arm'], r['k'], r['global_batch']) for r in results if r['status'] == 'oom'}
    ):
        lines.append(f'| {arm} | {k} | {b} | OOM | | | OOM |')

    lines += [
        '',
        '## Cost in B (fixed k) and cost in k (fixed B)',
        '',
        '| arm | B-slope k=2 | B-slope k=3 | t(k=3)/t(k=2) per B | '
        'mem(k=3)/mem(k=2) per B |',
        '|---|---|---|---|---|',
    ]
    for arm in sorted({a for a, _, _ in agg}):
        cells = []
        for k in (2, 3):
            pts = sorted(
                (b, agg[(arm, kk, b)]['step_ms'])
                for (aa, kk, b) in agg
                if aa == arm and kk == k
            )
            cells.append(f'{slope(*zip(*pts)):.2f}' if len(pts) >= 2 else '–')
        bs = sorted(
            b for (aa, kk, b) in agg if aa == arm and kk == 2 and (arm, 3, b) in agg
        )
        tr = (
            ', '.join(
                f"{b}: {agg[(arm, 3, b)]['step_ms'] / agg[(arm, 2, b)]['step_ms']:.2f}"
                for b in bs
            )
            or '–'
        )
        mr = (
            ', '.join(
                f"{b}: {agg[(arm, 3, b)]['peak_mem_mib'] / agg[(arm, 2, b)]['peak_mem_mib']:.2f}"
                for b in bs
            )
            or '–'
        )
        lines.append(f'| {arm} | {cells[0]} | {cells[1]} | {tr} | {mr} |')
    lines += [
        '',
        'B-slope: log-log slope of step time against global B over the '
        'sweep (1 = linear, 2 = quadratic); small B is launch-bound and '
        'flattens it. step − encoder: arm minus encoder_only at the same '
        "(k, B), i.e. the objective's forward + backward cost end to end, "
        'including the predictor heads that volume and pairwise train and '
        'infonce does not have.',
    ]
    return '\n'.join(lines)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def write_outputs(cli: Any, device: torch.device, results: List[Dict[str, Any]]) -> str:
    """Write the JSON and markdown summary (rank 0); returns the table."""
    table = summarize(results)
    if cli.out:
        Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
        Path(cli.out).write_text(
            json.dumps(
                {
                    'cli': vars(cli),
                    'device': torch.cuda.get_device_name(device),
                    'results': results,
                },
                indent=2,
            )
        )
    if cli.emit_markdown:
        Path(cli.emit_markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(cli.emit_markdown).write_text(table + '\n')
    return table


def main() -> None:
    cli = parser.parse_args()
    rank, world, device = setup()
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format='%(asctime)s %(message)s',
    )
    torch.backends.cuda.matmul.allow_tf32 = False  # match training defaults
    configs = {2: cli.k2_config, 3: cli.k3_config}
    if cli.emulate_world > 1:
        if world > 1:
            raise ValueError('--emulate-world is single-process; do not use torchrun')
        emulate_gathers(cli.emulate_world)
    # Ranks the encoder is split over: real (torchrun) or emulated.
    split = world * cli.emulate_world
    results: List[Dict[str, Any]] = []

    for k in cli.k:
        _, base_cfg = load_yaml(configs[k])
        for B in cli.batches:
            if B % split:
                raise ValueError(
                    f'global batch {B} not divisible by {split} (real x emulated ranks)'
                )
            per_rank = B // split
            for seed in cli.seeds:
                batches: Optional[List[Batch]] = None
                for arm in cli.arms:
                    if arm == 'graphclip' and k != 2:
                        continue
                    cfg = (
                        load_yaml(cli.graphclip_config)[1]
                        if arm == 'graphclip'
                        else base_cfg
                    )
                    rec: Dict[str, Any] = {
                        'arm': arm,
                        'k': k,
                        'global_batch': B,
                        'per_rank_batch': per_rank,
                        'world_size': split,
                        'emulated': cli.emulate_world > 1,
                        'checkpoint_encoder': cli.checkpoint_encoder,
                        'seed': seed,
                        'data': cli.data,
                    }
                    model = criterion = optimizer = None
                    try:
                        args = build_args(cfg, arm, per_rank, cli.fixed_graph_layers)
                        if batches is None:
                            batches = (
                                synthetic_batches(
                                    args,
                                    cli.n_cached_batches,
                                    per_rank,
                                    tuple(cli.syn_nodes),
                                    cli.syn_degree,
                                    seed * 1000 + rank,
                                )
                                if cli.data == 'synthetic'
                                else real_batches(
                                    args, cli.n_cached_batches, per_rank, seed, rank
                                )
                            )
                        rec |= batch_stats(batches)
                        torch.manual_seed(seed)
                        model, criterion, optimizer = build_model(
                            args, arm, device, world, cli.checkpoint_encoder
                        )
                        rec['trainable_params'] = sum(
                            p.numel()
                            for grp in optimizer.param_groups
                            for p in grp['params']
                        )
                        rec['graph_num_layers'] = args.graph_num_layers
                        rec['graph_hidden_dim'] = args.graph_hidden_dim
                        rec |= time_config(
                            model,
                            criterion,
                            optimizer,
                            args,
                            batches,
                            device,
                            cli.warmup,
                            cli.iters,
                        )
                        rec['status'] = 'ok'
                    except torch.cuda.OutOfMemoryError:
                        rec['status'] = 'oom'
                        if is_dist():
                            # Under DDP the other ranks are blocked inside a
                            # collective this rank will never join, so the job
                            # cannot continue: it would only hang until the
                            # NCCL watchdog aborts it. Record and exit now.
                            # Everything measured so far is already on disk.
                            logging.error(
                                f'OOM rank={rank} arm={arm} k={k} B={B} '
                                f'per_rank={per_rank}; '
                                f'{torch.cuda.max_memory_allocated(device) / 2**20:.0f} '
                                'MiB allocated at peak. Exiting the DDP job.'
                            )
                            if rank == 0:
                                write_outputs(cli, device, results + [rec])
                            for h in logging.getLogger().handlers:
                                h.flush()
                            os._exit(3)
                    except (FileNotFoundError, ImportError) as e:
                        if arm != 'graphclip':
                            raise
                        # Vendored GraphCLIP repo missing (GRAPHCLIP_ROOT).
                        rec['status'] = 'unavailable'
                        if rank == 0:
                            logging.warning(f'graphclip arm skipped: {e}')
                    del model, criterion, optimizer
                    gc.collect()
                    torch.cuda.empty_cache()
                    if is_dist():
                        # All ranks must agree: if any OOMed, all report OOM.
                        flag = torch.tensor(
                            [rec['status'] != 'ok'], device=device, dtype=torch.int
                        )
                        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                        if flag.item() and rec['status'] == 'ok':
                            rec['status'] = 'oom'
                    if rank == 0:
                        msg = (
                            f"step {rec['step_ms']:.1f} ms "
                            f"[{rec['step_ms_p25']:.1f}, {rec['step_ms_p75']:.1f}] "
                            f"(fwd {rec['forward_ms']:.1f} loss {rec['loss_ms']:.2f} "
                            f"bwd {rec['backward_ms']:.1f} opt {rec['optim_ms']:.1f}) "
                            f"peak {rec['peak_mem_mib']:.0f} MiB"
                            if rec['status'] == 'ok'
                            else rec['status'].upper()
                        )
                        logging.info(f'arm={arm:12s} k={k} B={B:6d} seed={seed}  {msg}')
                    results.append(rec)
                    if rank == 0:
                        # Rewrite after every configuration, so a crash later
                        # in the sweep keeps everything measured before it.
                        write_outputs(cli, device, results)
                del batches
                gc.collect()

    if rank == 0:
        if results and 'nodes_per_subgraph' in results[0]:
            logging.info(
                'subgraph stats (rank 0, cached batches): nodes %.1f, edges %.1f',
                results[0]['nodes_per_subgraph'],
                results[0]['edges_per_subgraph'],
            )
        print('\n' + write_outputs(cli, device, results))
    if is_dist():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
