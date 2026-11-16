"""Measure the per-step cost of volumetric+SIGReg vs contrastive alignment.

The claim under test, for a global batch B, embedding width d, and k modalities:

    volume + SIGReg   O(B d k^3)  +  O(k B M (d + T))      linear in B
    InfoNCE           O(C(k,2) B^2 d)                      quadratic in B and k

Both losses are fed identical synthetic embeddings, so the encoder is out of the
picture and what is timed is the alignment term alone. --with-encoder adds a
stand-in MLP of a comparable parameter count to show where the loss sits in a
real step; the loss is a small share of step time until B is large, and saying
so is part of the claim rather than an objection to it.

What is reported per configuration: forward+backward wall time (CUDA events,
after warmup), peak allocated memory, and a log-log slope of time against B.
Slope ~1 is linear, ~2 is quadratic. Small B is dominated by kernel-launch
overhead and flattens the slope, so the fit uses the upper half of the sweep and
the full curve is printed for inspection.

Distributed note. Under DDP with world size W and per-rank batch B_loc, the
global batch is B = W B_loc. InfoNCE must all-gather every modality and every
rank then computes the full B x B logit matrix, so per-rank contrastive cost is
quadratic in the GLOBAL batch and does not fall as ranks are added. The volume
term is per sample and needs no gather, so per-rank cost stays O(B_loc d k^3):
adding ranks divides it. --world-size models this without needing W GPUs.

    python tools/bench_alignment_cost.py --sweep batch --k 2 3
    python tools/bench_alignment_cost.py --sweep modalities --batch 4096
    python tools/bench_alignment_cost.py --sweep batch --k 2 --with-encoder
"""

import argparse
import json
import math
import os
import resource
import time
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn.functional import normalize

parser = argparse.ArgumentParser(
    description='Per-step cost of volumetric vs contrastive alignment.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--sweep', choices=['batch', 'modalities'], default='batch')
parser.add_argument(
    '--batches',
    type=int,
    nargs='+',
    default=[256, 512, 1024, 2048, 4096, 8192, 16384],
    help='Global batch sizes for --sweep batch.',
)
parser.add_argument('--batch', type=int, default=4096, help='B for --sweep modalities.')
parser.add_argument('--k', type=int, nargs='+', default=[2, 3])
parser.add_argument(
    '--max-k', type=int, default=6, help='Largest k for --sweep modalities.'
)
parser.add_argument('--dim', type=int, default=768, help='Shared embedding width d.')
parser.add_argument('--sigreg-slices', type=int, default=128, help='M.')
parser.add_argument('--sigreg-quad', type=int, default=17, help='T.')
parser.add_argument('--world-size', type=int, default=1, help='Model DDP with W ranks.')
parser.add_argument(
    '--device',
    choices=['auto', 'cuda', 'cpu'],
    default='auto',
    help='Where to run. CPU reaches batch sizes device memory forbids, but its '
    'timings do NOT transfer: there is no kernel-launch overhead and the memory '
    'hierarchy differs, so the crossover B* measured on CPU says nothing about '
    'the GPU one. Use a CPU sweep for the asymptotic exponents only.',
)
parser.add_argument(
    '--threads', type=int, default=0, help='torch.set_num_threads for --device cpu.'
)
parser.add_argument('--with-encoder', action='store_true')
parser.add_argument('--encoder-width', type=int, default=1024)
parser.add_argument('--encoder-depth', type=int, default=6)
parser.add_argument('--warmup', type=int, default=5)
parser.add_argument('--iters', type=int, default=20, help='Maximum timed iterations.')
parser.add_argument(
    '--min-iters', type=int, default=3, help='Floor on timed iterations.'
)
parser.add_argument(
    '--target-ms',
    type=float,
    default=500.0,
    help='Per-configuration timing budget. After warmup one iteration is timed '
    'and the count is chosen to fill this budget, capped by --iters. Keeps a '
    'sweep that reaches B ~ 1e5 from spending minutes on its largest points, '
    'where the per-iteration time is long and the relative launch overhead is '
    'already negligible. Set 0 to always use --iters.',
)
parser.add_argument('--out', type=str, default='', help='Write raw results as JSON.')
parser.add_argument(
    '--emit-latex', type=str, default='', help='Write an appendix table.'
)
parser.add_argument(
    '--emit-figure',
    type=str,
    default='',
    help='Write the log-log figure. The extension is ignored: both a .pdf and '
    'a .png are written from the same render, so the vector copy for the paper '
    'and the raster copy for quick viewing cannot drift apart.',
)
parser.add_argument(
    '--figure-facecolor',
    type=str,
    default='white',
    help="Figure and axes background. 'white' matches the ICLR page. 'none' is "
    'transparent, which is what a coloured or dark template wants; any '
    'matplotlib colour also works.',
)
parser.add_argument(
    '--no-split-figures',
    action='store_true',
    help='Write only the combined figure. By default each panel is also '
    'written on its own (<stem>_time, <stem>_memory, or <stem>_k<K>_<panel> '
    'when the sweep covers several k) so the time and memory results can be '
    'placed and cited independently.',
)
parser.add_argument(
    '--use-repo',
    action='store_true',
    help="Time the repo's own SIGReg and squared_volume instead of the reference "
    'implementations below. Use this for any number that goes in the paper: the '
    'reference SIGReg here is a plausible implementation, not necessarily yours, '
    'and it dominates the volumetric arm.',
)
parser.add_argument(
    '--fit-from',
    type=int,
    default=0,
    help='Smallest B included in the exponent fit. 0 uses the upper half of '
    'the sweep. Fitting across the launch-overhead regime biases the exponent '
    'downward, so report the window alongside the number.',
)
parser.add_argument(
    '--per-sample',
    action='store_true',
    help='Plot time and memory per sample (divided by B). A linear term is '
    'then a flat line and a quadratic one has slope 1, which is easier to '
    'read than distinguishing slope 1 from slope 2.',
)
parser.add_argument(
    '--fast-det',
    action='store_true',
    help='Use squared_volume_fast for the determinant term: the same quantity '
    'computed as k(k-1)/2 pairwise cosine reductions instead of a batched '
    '(k,d)@(d,k) matmul. Run the sweep with and without to measure the gap.',
)
parser.add_argument(
    '--components',
    action='store_true',
    help='Time the determinant term and SIGReg separately, so the volumetric '
    "arm's cost can be attributed. Adds two series: volume_only and sigreg.",
)


# --------------------------------------------------------------------------
# Reference implementations. These mirror tgfm/models/losses/volumeloss.py and
# GraphCLIP models/dp.py so the benchmark runs standalone; --use-repo swaps in
# the real modules when the repo is importable.
# --------------------------------------------------------------------------


def squared_volume(
    embeddings: List[Tensor], predictions: List[Tensor], anchor: int
) -> Tensor:
    """Det G_i^(a) per sample. Anchor column live, others detached."""
    columns = [
        normalize(predictions[anchor] if b == anchor else z.detach(), dim=-1, eps=1e-8)
        for b, z in enumerate(embeddings)
    ]
    A = torch.stack(columns, dim=-1)  # (B, d, k)
    G = A.transpose(-2, -1) @ A  # (B, k, k)
    return torch.linalg.det(G)


class SIGRegRef(torch.nn.Module):
    """Sliced Epps-Pulley statistic against an isotropic Gaussian.

    M random unit directions, empirical characteristic function at T quadrature
    points, squared deviation from exp(-t^2/2). Cost is O(B M d) for the
    projection plus O(B M T) for the characteristic function: linear in B.
    """

    def __init__(
        self, num_slices: int, num_quad_points: int, t_max: float = 4.0
    ) -> None:
        super().__init__()
        self.num_slices = num_slices
        self.register_buffer('t', torch.linspace(-t_max, t_max, num_quad_points))
        self.dt = 2 * t_max / max(1, num_quad_points - 1)

    def forward(self, z: Tensor) -> Tensor:
        B, d = z.shape
        w = normalize(
            torch.randn(d, self.num_slices, device=z.device, dtype=z.dtype), dim=0
        )
        s = z @ w  # (B, M)           O(B M d)
        s = (s - s.mean(0, keepdim=True)) / s.std(0, keepdim=True).clamp_min(1e-6)
        ts = self.t.to(z.dtype).view(1, 1, -1)
        arg = s.unsqueeze(-1) * ts  # (B, M, T)   O(B M T)
        re = torch.cos(arg).mean(0)
        im = torch.sin(arg).mean(0)
        target = torch.exp(-0.5 * self.t.to(z.dtype) ** 2).view(1, -1)
        return (((re - target) ** 2 + im**2) * self.dt).sum(-1).mean()


def squared_volume_fast(
    embeddings: List[Tensor], predictions: List[Tensor], anchor: int
) -> Tensor:
    """Det G_i^(a) without ever forming the Gram matrix.

    The columns are unit-normalised, so diag(G) = 1 exactly and the matrix is
    determined by its k(k-1)/2 off-diagonal cosines. Expanding the determinant
    in those,

        k = 2:  det G = 1 - c01^2
        k = 3:  det G = 1 - c01^2 - c02^2 - c12^2 + 2 c01 c02 c12   (GRAM Eq. 13)

    turns the batched (k,d)@(d,k) matmul into k(k-1)/2 reductions over d. Same
    arithmetic, but the matmul form asks cuBLAS for B independent products with
    a 2x2 or 3x3 output, which is a regime its batched path handles very badly.
    Gradient semantics are unchanged: the anchor column is live and the rest are
    detached exactly as before, so every cosine touching the anchor carries
    gradient and the others are constants.
    """
    cols = [
        normalize(predictions[anchor] if b == anchor else z.detach(), dim=-1, eps=1e-8)
        for b, z in enumerate(embeddings)
    ]
    k = len(cols)
    if k == 2:
        c = (cols[0] * cols[1]).sum(-1)
        return 1.0 - c * c
    if k == 3:
        c01 = (cols[0] * cols[1]).sum(-1)
        c02 = (cols[0] * cols[2]).sum(-1)
        c12 = (cols[1] * cols[2]).sum(-1)
        return 1.0 - c01**2 - c02**2 - c12**2 + 2.0 * c01 * c02 * c12
    A = torch.stack(cols, dim=-1)
    return torch.linalg.det(A.transpose(-2, -1) @ A)


def resolve_impls(use_repo: bool, fast_det: bool = False):
    """(squared_volume, SIGReg class, source label).

    With --use-repo the repo's own modules are imported, so the measurement
    reflects the code that actually trains. Without it the reference
    implementations in this file are used and the run is self-contained.
    """
    if not use_repo:
        det = squared_volume_fast if fast_det else squared_volume
        return det, SIGRegRef, f'reference (this file), det={det.__name__}'

    from tgfm.models import losses as _losses  # noqa: F401  (package import guard)
    from tgfm.models.losses import volumeloss as repo_vol
    from tgfm.models.sigreg import SIGReg as RepoSIGReg

    src = 'repo'
    if not fast_det:
        det = repo_vol.squared_volume
    elif hasattr(repo_vol, 'squared_volume_fast'):
        # Prefer the repo's own fast path so the run measures the edited file
        # rather than this benchmark's copy of it.
        det = repo_vol.squared_volume_fast
    else:
        det = squared_volume_fast
        src = 'repo SIGReg + benchmark det (volumeloss has no squared_volume_fast)'
    return det, RepoSIGReg, f'{src} (tgfm.models), det={det.__name__}'


def volume_loss(
    z: List[Tensor],
    pred: List[Tensor],
    sigreg,
    lam: float,
    world_size: int,
    det_fn=None,
    with_sigreg: bool = True,
    with_align: bool = True,
) -> Tensor:
    """Volume alignment + per-modality SIGReg, matching LeGramJEPA Eq. 8.

    world_size > 1 replicates what all_gather does to the SIGReg input size: the
    regularizer sees the global batch while the volume term stays per-rank.
    """
    k = len(z)
    det_fn = det_fn or squared_volume
    total = torch.zeros((), device=z[0].device, dtype=z[0].dtype)
    if with_align:
        dets = [det_fn(z, pred, a) for a in range(k)]
        total = total + (1.0 - lam) * torch.stack([d.mean() for d in dets]).mean()
    if with_sigreg:
        reg = torch.zeros((), device=z[0].device, dtype=z[0].dtype)
        for za in z:
            zg = za if world_size == 1 else za.repeat(world_size, 1)
            reg = reg + sigreg(zg)
        total = total + (lam / k) * reg
    return total


def infonce_loss(z: List[Tensor], logit_scale: Tensor, world_size: int) -> Tensor:
    """Symmetric InfoNCE over every modality pair, on the gathered batch.

    Every rank computes the full global B x B matrix for each of C(k,2) pairs,
    which is what makes the per-rank cost quadratic in the global batch.
    """
    k = len(z)
    gathered = [za if world_size == 1 else za.repeat(world_size, 1) for za in z]
    units = [normalize(za, dim=-1) for za in gathered]
    B = units[0].shape[0]
    target = torch.arange(B, device=units[0].device)
    scale = logit_scale.exp()
    ce = torch.nn.functional.cross_entropy
    total = torch.zeros((), device=units[0].device, dtype=units[0].dtype)
    pairs = 0
    for a in range(k):
        for b in range(a + 1, k):
            logits = scale * units[a] @ units[b].t()  # (B, B)   O(B^2 d)
            total = total + 0.5 * (ce(logits, target) + ce(logits.t(), target))
            pairs += 1
    return total / max(1, pairs)


def make_encoder(in_dim: int, width: int, depth: int, out_dim: int) -> torch.nn.Module:
    layers: List[torch.nn.Module] = []
    prev = in_dim
    for _ in range(depth):
        layers += [
            torch.nn.Linear(prev, width),
            torch.nn.BatchNorm1d(width),
            torch.nn.GELU(),
        ]
        prev = width
    layers.append(torch.nn.Linear(prev, out_dim))
    return torch.nn.Sequential(*layers)


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


def time_step(
    objective: str,
    B: int,
    k: int,
    d: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    """Mean ms and peak MiB for one forward+backward of the given objective.

    objective is one of:
        volume       determinant term + SIGReg, the full LeGramJEPA alignment
        volume_only  determinant term alone
        sigreg       SIGReg alone
        infonce      pairwise-sum symmetric InfoNCE
    """
    torch.manual_seed(0)
    det_fn, sigreg_cls, _ = resolve_impls(args.use_repo, args.fast_det)
    sigreg = sigreg_cls(
        num_slices=args.sigreg_slices,
        num_quad_points=args.sigreg_quad,
        t_max=4.0,
    ).to(device)
    logit_scale = torch.tensor(math.log(1 / 0.07), device=device, requires_grad=True)

    if args.with_encoder:
        encoders = [
            make_encoder(d, args.encoder_width, args.encoder_depth, d).to(device)
            for _ in range(k)
        ]
        predictors = [
            make_encoder(d, args.encoder_width, 2, d).to(device) for _ in range(k)
        ]
        params = [p for m in encoders + predictors for p in m.parameters()] + [
            logit_scale
        ]
        inputs = [torch.randn(B, d, device=device) for _ in range(k)]
    else:
        encoders = predictors = []
        leaves = [
            torch.randn(B, d, device=device, requires_grad=True) for _ in range(k)
        ]
        pred_leaves = [
            torch.randn(B, d, device=device, requires_grad=True) for _ in range(k)
        ]
        params = leaves + pred_leaves + [logit_scale]
        inputs = None

    def one_step() -> None:
        for p in params:
            p.grad = None
        if args.with_encoder:
            z = [enc(x) for enc, x in zip(encoders, inputs)]
            pred = [pr(za) for pr, za in zip(predictors, z)]
        else:
            z, pred = leaves, pred_leaves
        if objective == 'infonce':
            loss = infonce_loss(z, logit_scale, args.world_size)
        else:
            loss = volume_loss(
                z,
                pred,
                sigreg,
                0.01,
                args.world_size,
                det_fn=det_fn,
                with_align=objective in ('volume', 'volume_only'),
                with_sigreg=objective in ('volume', 'sigreg'),
            )
        loss.backward()

    for _ in range(args.warmup):
        one_step()

    def timed(n: int) -> float:
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(n):
            one_step()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / n

    def timed_cpu(n: int) -> float:
        t0 = time.perf_counter()
        for _ in range(n):
            one_step()
        return (time.perf_counter() - t0) * 1000 / n

    clock = timed if device.type == 'cuda' else timed_cpu
    iters = args.iters
    if args.target_ms > 0:
        # One probe iteration sets the count, so small B still gets enough
        # repetitions to average out launch jitter while large B does not run
        # for minutes. The probe is discarded.
        probe = clock(1)
        iters = max(
            args.min_iters, min(args.iters, int(args.target_ms / max(probe, 1e-6)))
        )

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        ms = clock(iters)
        peak = torch.cuda.max_memory_allocated() / 2**20
    else:
        # No CPU equivalent of max_memory_allocated. ru_maxrss is the process
        # high-water mark, so it is only attributable to this configuration
        # while the sweep runs in increasing B and each configuration is the
        # largest so far -- which is how --sweep batch orders them. It also
        # includes the interpreter and any cached allocator blocks, so read it
        # as an upper bound rather than the tensor footprint.
        ms = clock(iters)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {'ms': ms, 'peak_mib': peak, 'iters': float(iters)}


def loglog_slope(xs: List[float], ys: List[float]) -> float:
    """Least-squares slope of log y against log x. NaN if fewer than 2 points."""
    if len(xs) < 2:
        return float('nan')
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den if den > 0 else float('nan')


def fit_exponent(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    """(exponent, 95% CI half-width, R^2) for the power law y = c x^p.

    Fitting log y against log x, so the exponent is the log-log slope. R^2 says
    whether a power law describes the data at all: a value well below 1 means
    the window still contains a regime change and the exponent is not
    meaningful. The CI is the usual t-free normal approximation, which is
    adequate for the 4-5 points a sweep of this shape provides.
    """
    n = len(xs)
    if n < 3:
        return float('nan'), float('nan'), float('nan')
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((a - mx) ** 2 for a in lx)
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    if sxx <= 0:
        return float('nan'), float('nan'), float('nan')
    p = sxy / sxx
    a = my - p * mx
    sse = sum((b - (a + p * x)) ** 2 for x, b in zip(lx, ly))
    sst = sum((b - my) ** 2 for b in ly)
    se = math.sqrt(sse / (n - 2) / sxx) if n > 2 else float('nan')
    return p, 1.96 * se, (1 - sse / sst if sst > 0 else float('nan'))


def crossover_B(xs: List[int], a: List[float], b: List[float]) -> float:
    """Batch size where series a and b cross, by log-log interpolation.

    Reported rather than hidden: the asymptotic ordering says nothing about
    which side of this point a given operating batch sits on.
    """
    prev = None
    for B, ya, yb in zip(xs, a, b):
        if ya <= 0 or yb <= 0:
            continue
        r = ya / yb
        if prev is not None and (prev[1] - 1.0) * (r - 1.0) < 0:
            B0, r0 = prev
            t = -math.log(r0) / (math.log(r) - math.log(r0))
            return math.exp(math.log(B0) + t * (math.log(B) - math.log(B0)))
        prev = (B, r)
    return float('nan')


def predicted_flops(objective: str, B: int, k: int, d: int, M: int, T: int) -> float:
    """Analytic MAC count for the alignment term (Props. 1-3 of the appendix)."""
    if objective == 'volume':
        return B * d * k**3 + k * B * M * (d + T)
    return (k * (k - 1) / 2) * B * B * d


def predicted_bytes(objective: str, B: int, k: int, d: int, M: int, T: int) -> float:
    """Analytic retained-scalar count, times 4 for fp32 (Remark on memory).

    The volumetric term keeps, per anchor, the k normalised columns and the
    stacked A, hence 2*B*d*k^2 over k anchors, plus SIGReg's (B, M, T)
    quadrature tensor. Those constants are what put the memory crossover near
    B = 9000 for k = 2, far above the B* = 270 of the time crossover.
    """
    if objective == 'volume':
        return 4.0 * (2 * B * d * k**2 + k * B * M * T)
    return 4.0 * ((k * (k - 1) / 2) * B * B + k * B * d)


# Predicted log-log slopes in B, from the appendix: time and memory are both
# linear for the volumetric term and quadratic for the contrastive one.
PREDICTED_SLOPE = {'volume': 1.0, 'volume_only': 1.0, 'sigreg': 1.0, 'infonce': 2.0}

# Categorical slots 1 and 2 of the reference palette; the adjacent pair was
# checked for CVD separation (protan dE 24.7, normal dE 33.6) before use.
SERIES_COLOR = {
    'volume': '#2a78d6',
    'infonce': '#eb6834',
    'volume_only': '#1baf7a',
    'sigreg': '#eda100',
}
SERIES_MARKER = {
    'volume': 'o',
    'infonce': 's',
    'volume_only': '^',
    'sigreg': 'D',
}
SERIES_LABEL = {
    'volume': 'volumetric + SIGReg',
    'infonce': 'pairwise-sum InfoNCE',
    'volume_only': 'determinant term only',
    'sigreg': 'SIGReg only',
}


def emit_latex(rows: List[Dict], args: argparse.Namespace, path: str) -> None:
    """Appendix table: measured time and memory, with the fitted exponents.

    Cells where an arm ran out of memory are printed as OOM rather than
    dropped: the batch at which the contrastive term stops and the volumetric
    one continues is a result, not a gap in the data. Exponents are fitted per
    objective over whatever batches that objective survived, so a truncated
    series still gets a number, and the fitting window is stated in the
    caption because fitting across the launch-overhead regime biases it down.
    """
    batch_rows = [r for r in rows if r['sweep'] == 'batch']
    if not batch_rows:
        return
    ks = sorted({r['k'] for r in batch_rows})

    def cell(k: int, B: int, o: str, field: str) -> str:
        m = [
            r for r in batch_rows if r['k'] == k and r['B'] == B and r['objective'] == o
        ]
        if not m:
            return r'\textsc{oom}'
        return f'{m[0][field]:.2f}' if field == 'ms' else f'{m[0][field]:.1f}'

    fit_from = args.fit_from
    window = f'$B \\geq {fit_from}$' if fit_from else 'the upper half of the sweep'
    lines = [
        '% generated by tools/bench_alignment_cost.py',
        '\\begin{table}[t]',
        '\\caption{Measured cost of the alignment term, forward and backward, on '
        'synthetic embeddings of width $d=' + str(args.dim) + '$, encoders excluded. '
        'Exponents are least-squares fits of $\\log t$ against $\\log B$ over '
        + window
        + ', where kernel-launch overhead no longer dominates; '
        'Propositions~\\ref{prop:nce} and~\\ref{prop:vol} predict $1$ for the '
        'volumetric term and $2$ for the contrastive one. \\textsc{oom} marks '
        'configurations that exceeded device memory.}',
        '\\label{tab:cost-measured}',
        '\\begin{center}',
        '\\begin{tabular}{rrrrrr}',
        '\\toprule',
        '& & \\multicolumn{2}{c}{\\bf time (ms)} & '
        '\\multicolumn{2}{c}{\\bf peak memory (MiB)} \\\\',
        '\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}',
        '\\multicolumn{1}{c}{\\bf $k$} & \\multicolumn{1}{c}{\\bf $B$} & volumetric & '
        'contrastive & volumetric & contrastive \\\\',
        '\\midrule',
    ]
    for k in ks:
        sub = sorted({r['B'] for r in batch_rows if r['k'] == k})
        for i, B in enumerate(sub):
            head = f'{k}' if i == 0 else ''
            lines.append(
                f'{head} & {B} & {cell(k, B, "volume", "ms")} & '
                f'{cell(k, B, "infonce", "ms")} & '
                f'{cell(k, B, "volume", "peak_mib")} & '
                f'{cell(k, B, "infonce", "peak_mib")} \\\\'
            )
        fitted = {}
        for o in ('volume', 'infonce'):
            pts = sorted(
                (r['B'], r['ms'], r['peak_mib'])
                for r in batch_rows
                if r['k'] == k and r['objective'] == o
            )
            lo = fit_from or (pts[len(pts) // 2][0] if pts else 0)
            sel = [q for q in pts if q[0] >= lo]
            if len(sel) < 3:
                fitted[(o, 'ms')] = fitted[(o, 'peak_mib')] = float('nan')
                continue
            xs = [q[0] for q in sel]
            fitted[(o, 'ms')] = fit_exponent(xs, [q[1] for q in sel])[0]
            fitted[(o, 'peak_mib')] = fit_exponent(xs, [q[2] for q in sel])[0]

        def fmt(v: float) -> str:
            return '--' if v != v else f'{v:.2f}'

        lines.append('\\cmidrule(lr){1-6}')
        lines.append(
            f' & exponent & {fmt(fitted[("volume", "ms")])} & '
            f'{fmt(fitted[("infonce", "ms")])} & '
            f'{fmt(fitted[("volume", "peak_mib")])} & '
            f'{fmt(fitted[("infonce", "peak_mib")])} \\\\'
        )
        if k != ks[-1]:
            lines.append('\\midrule')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{center}', '\\end{table}', '']
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines))
    print(f'wrote {path}')


def _panels(per_sample: bool) -> List[Tuple[str, str, str]]:
    """(field, y label, filename suffix) for each measure."""
    if per_sample:
        return [
            ('ms', 'forward + backward\n($\\mu$s / sample)', 'time'),
            ('peak_mib', 'peak memory\n(KiB / sample)', 'memory'),
        ]
    return [
        ('ms', 'forward + backward (ms)', 'time'),
        ('peak_mib', 'peak memory (MiB)', 'memory'),
    ]


def _draw_panel(
    ax, batch_rows, k, field, ylabel, objs, args, label_series, show_x, show_k
):
    """One log-log axes: every series for one k and one measure.

    Axis limits come from the data alone. The reference guides are drawn after
    the limits are fixed and are clipped to them, because a slope-2 guide
    spanning the whole sweep covers twice the decades the data does and would
    otherwise flatten every curve into the middle of the frame.
    """
    per_sample = getattr(args, 'per_sample', False)

    def scaled(xs, ys):
        if not per_sample:
            return ys
        f = 1000.0 if field == 'ms' else 1024.0
        return tuple(y * f / x for x, y in zip(xs, ys))

    def series(o):
        return sorted(
            (r['B'], r[field])
            for r in batch_rows
            if r['objective'] == o and r['k'] == k
        )

    lo, hi = float('inf'), 0.0
    for o in objs:
        pts = series(o)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ys = scaled(xs, ys)
        lo, hi = min(lo, min(ys)), max(hi, max(ys))
        ax.plot(
            xs,
            ys,
            color=SERIES_COLOR[o],
            linewidth=2.0,
            markersize=5,
            marker=SERIES_MARKER.get(o, 'o'),
            linestyle='-',
            label=SERIES_LABEL[o] if label_series else '_nolegend_',
        )
    if hi <= 0:
        return False

    # A series that stops short of the panel's largest batch ran out of memory
    # there. Saying so on the plot makes the strongest part of the comparison
    # explicit instead of leaving a line that merely appears to end.
    b_max = max(r['B'] for r in batch_rows if r['k'] == k)
    for o in objs:
        pts = series(o)
        if not pts or pts[-1][0] >= b_max:
            continue
        ex, ey = pts[-1][0], scaled([pts[-1][0]], [pts[-1][1]])[0]
        ax.annotate(
            'OOM',
            xy=(ex, ey),
            xytext=(6, 2),
            textcoords='offset points',
            fontsize=7,
            color=SERIES_COLOR[o],
            fontweight='bold',
            annotation_clip=False,
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(lo / 1.6, hi * 1.6)

    # Guides over the top decade only: enough to read the local slope, short
    # enough not to reach for the bottom of the frame.
    for o, ref in (('volume', 1.0), ('infonce', 2.0)):
        pts = series(o)
        if len(pts) < 2:
            continue
        bx = pts[-1][0]
        by = scaled([bx], [pts[-1][1]])[0]
        e = ref - 1.0 if per_sample else ref
        gx = [max(pts[0][0], bx / 10.0), bx]
        ax.plot(
            gx,
            [by * (x / bx) ** e for x in gx],
            color='#8c8c86',
            linewidth=0.9,
            linestyle=':',
            zorder=0,
            clip_on=True,
            label='_nolegend_',
        )

    if show_x:
        ax.set_xlabel('batch size $B$')
    ax.set_ylabel(ylabel)
    if show_k:
        ax.annotate(
            f'$k = {k}$',
            xy=(0.03, 0.93),
            xycoords='axes fraction',
            fontsize=9,
            color='#0b0b0b',
            fontweight='bold',
        )
    ax.grid(True, which='major', color='#e4e4e0', linewidth=0.8)
    ax.grid(True, which='minor', color='#f0f0ec', linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#c8c8c3')
    ax.tick_params(colors='#52514e', labelsize=8)
    ax.xaxis.label.set_color('#52514e')
    ax.yaxis.label.set_color('#52514e')
    return True


def _save(fig, stem: str, written: List[str]) -> None:
    """Vector copy for \\includegraphics and a raster copy from the same render.

    facecolor is forced at save time as well as at creation: savefig otherwise
    re-applies the rcParam rather than the figure's own colour, which is how a
    tinted background survives a white figure.
    """
    for ext, kwargs in (('.pdf', {}), ('.png', {'dpi': 200})):
        out = stem + ext
        fig.savefig(
            out,
            bbox_inches='tight',
            facecolor=fig.get_facecolor(),
            edgecolor='none',
            **kwargs,
        )
        written.append(out)


def emit_figure(rows: List[Dict], args: argparse.Namespace, path: str) -> None:
    """Log-log figures of time and peak memory against B.

    Writes the combined figure (columns are the two measures, rows are k) and,
    unless --no-split-figures, each panel again on its own. A paper usually
    wants the two separately: the time result and the memory result support
    different claims and belong at different points in the text, and a
    \\subref into a single graphic cannot be cited independently.

    Faceting by k rather than stacking every series in one pair of axes keeps
    each panel to at most four lines. Two panels rather than two y-axes on one:
    the measures have different units, and a dual axis would make their
    crossing point an artefact of the scaling.

    Background is white by default so the figure sits on the page without a
    visible panel behind it; --figure-facecolor none makes it transparent,
    which is what a coloured or dark template wants.
    """
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Type 42 (TrueType) rather than matplotlib's default Type 3 in PDF.
        # Type 3 fonts are rejected by several submission systems, arXiv's
        # among them, and the failure shows up at upload rather than here.
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
    except ImportError:
        print('matplotlib not available; skipping figure')
        return

    batch_rows = [r for r in rows if r['sweep'] == 'batch']
    if not batch_rows:
        return
    ks = sorted({r['k'] for r in batch_rows})
    objs = [o for o in SERIES_COLOR if any(r['objective'] == o for r in batch_rows)]
    per_sample = getattr(args, 'per_sample', False)
    panels = _panels(per_sample)
    face = getattr(args, 'figure_facecolor', 'white')
    stem = os.path.splitext(path)[0]
    written: List[str] = []

    nrow, ncol = len(ks), len(panels)
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(7.2, 2.7 * nrow),
        constrained_layout=True,
        facecolor=face,
        squeeze=False,
    )
    for ri, k in enumerate(ks):
        for ci, (field, ylabel, _suffix) in enumerate(panels):
            ax = axes[ri][ci]
            ax.set_facecolor(face)
            _draw_panel(
                ax,
                batch_rows,
                k,
                field,
                ylabel,
                objs,
                args,
                label_series=(ri == 0 and ci == 0),
                show_x=(ri == nrow - 1),
                show_k=(ncol > 1 and ci == 0 and nrow > 1),
            )
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='outside upper center',
        ncol=min(4, len(labels)),
        frameon=False,
        fontsize=8,
        labelcolor='#0b0b0b',
    )
    _save(fig, stem, written)
    plt.close(fig)

    if not getattr(args, 'no_split_figures', False):
        multi_k = len(ks) > 1
        for k in ks:
            for field, ylabel, suffix in panels:
                sub, ax = plt.subplots(
                    figsize=(3.6, 2.9), constrained_layout=True, facecolor=face
                )
                ax.set_facecolor(face)
                ok = _draw_panel(
                    ax,
                    batch_rows,
                    k,
                    field,
                    ylabel,
                    objs,
                    args,
                    label_series=True,
                    show_x=True,
                    show_k=multi_k,
                )
                if not ok:
                    plt.close(sub)
                    continue
                h, lb = ax.get_legend_handles_labels()
                # Outside the axes: four labels inside a 3.6in panel would sit
                # on top of the curves they describe. constrained_layout
                # reserves the strip, so the axes keeps its full height.
                sub.legend(
                    h,
                    lb,
                    loc='outside upper center',
                    ncol=2,
                    frameon=False,
                    fontsize=7,
                    labelcolor='#0b0b0b',
                )
                part = f'{stem}_k{k}_{suffix}' if multi_k else f'{stem}_{suffix}'
                _save(sub, part, written)
                plt.close(sub)

    print('wrote ' + ', '.join(written))


def main() -> None:
    args = parser.parse_args()
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    if device.type != 'cuda':
        if args.threads:
            torch.set_num_threads(args.threads)
        print(
            'RUNNING ON CPU. Timings are not comparable to GPU: no kernel-launch\n'
            'overhead, different memory hierarchy, different BLAS. The exponents\n'
            'still measure the growth rate, but the crossover B* and every\n'
            'absolute time belong to this device only.\n'
            f'threads={torch.get_num_threads()}  '
            'memory column is process peak RSS, an upper bound.\n'
        )
    d, M, T = args.dim, args.sigreg_slices, args.sigreg_quad
    _, _, impl_label = resolve_impls(args.use_repo, args.fast_det)
    print(
        f'device={device} d={d} M={M} T={T} world_size={args.world_size} '
        f'encoder={args.with_encoder}\n'
        f'implementations: {impl_label}\n'
    )
    rows = []

    if args.sweep == 'batch':
        for k in args.k:
            print(f'=== k = {k} modalities ===')
            print(
                f'{"B":>7} {"volume ms":>10} {"nce ms":>10} {"speedup":>8} '
                f'{"vol MiB":>9} {"nce MiB":>9} {"pred ratio":>11}'
            )
            objs = ['volume', 'infonce']
            if args.components:
                objs = ['volume', 'volume_only', 'sigreg', 'infonce']
            per_obj: Dict[str, List[float]] = {o: [] for o in objs}
            per_batch: Dict[str, List[int]] = {o: [] for o in objs}
            for B in args.batches:
                res = {}
                for o in objs:
                    # Per objective, so an arm that OOMs does not delete the
                    # other's data point. The batch where the contrastive arm
                    # dies while the volumetric one keeps going is itself a
                    # result, and it is lost if both are skipped together.
                    try:
                        res[o] = time_step(o, B, k, d, args, device)
                    except (
                        torch.cuda.OutOfMemoryError,
                        MemoryError,
                        RuntimeError,
                    ) as exc:
                        # CPU exhaustion surfaces as MemoryError or a RuntimeError
                        # naming the allocation; anything else should not be
                        # swallowed as an out-of-memory result.
                        if (
                            isinstance(exc, RuntimeError)
                            and 'alloc' not in str(exc).lower()
                        ):
                            raise
                        res[o] = None
                    else:
                        per_obj[o].append(res[o]['ms'])
                        per_batch[o].append(B)
                        rows.append(
                            {'sweep': 'batch', 'objective': o, 'B': B, 'k': k, **res[o]}
                        )
                    finally:
                        # Each configuration allocates its own leaves; at the
                        # top of the sweep those are gigabytes, and leaving
                        # them cached fragments the next configuration.
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                if res.get('volume') is None and res.get('infonce') is None:
                    print(f'{B:>7} {"OOM in both arms":>50}')
                    continue

                def cell(o, f, spec):
                    return format(res[o][f], spec) if res.get(o) else 'OOM'

                ratio = predicted_flops('infonce', B, k, d, M, T) / predicted_flops(
                    'volume', B, k, d, M, T
                )
                speed = (
                    f'{res["infonce"]["ms"] / res["volume"]["ms"]:.2f}x'
                    if res.get('infonce') and res.get('volume')
                    else '-'
                )
                print(
                    f'{B:>7} {cell("volume", "ms", ".3f"):>10} '
                    f'{cell("infonce", "ms", ".3f"):>10} {speed:>8} '
                    f'{cell("volume", "peak_mib", ".1f"):>9} '
                    f'{cell("infonce", "peak_mib", ".1f"):>9} {ratio:>10.2f}x'
                )
                if args.components:
                    print(
                        f'{"":>7} {"components:":>10} det='
                        f'{cell("volume_only", "ms", ".3f")}ms '
                        f'sigreg={cell("sigreg", "ms", ".3f")}ms'
                    )
            batches_ok = per_batch['volume']
            fit_from = args.fit_from or (
                batches_ok[len(batches_ok) // 2] if batches_ok else 0
            )
            print(
                f'\n  fitted exponent of y = c B^p, over B >= {fit_from} '
                f'(below that, launch overhead dominates and flattens the fit)'
            )
            print(
                f'  {"":>11} {"exponent(time)":>17} {"R^2":>7} '
                f'{"exponent(mem)":>15} {"predicted":>10} {"verdict":>8}'
            )
            for o in per_obj:
                xs_all = per_batch[o]
                mem_all = [
                    r['peak_mib']
                    for r in rows
                    if r['sweep'] == 'batch' and r['k'] == k and r['objective'] == o
                ]
                sel = [i for i, B in enumerate(xs_all) if B >= fit_from]
                if len(sel) < 3:
                    continue
                xs = [xs_all[i] for i in sel]
                p_t, ci_t, r2_t = fit_exponent(xs, [per_obj[o][i] for i in sel])
                p_m, _, _ = fit_exponent(xs, [mem_all[i] for i in sel])
                pred = PREDICTED_SLOPE.get(o, float('nan'))
                ok = (
                    '-' if pred != pred else ('ok' if abs(p_t - pred) < 0.25 else 'OFF')
                )
                print(
                    f'  {o:>11} {p_t:>10.2f} +/- {ci_t:<4.2f} {r2_t:>7.4f} '
                    f'{p_m:>15.2f} {pred:>10.1f} {ok:>8}'
                )
            if 'volume' in per_obj and 'infonce' in per_obj:
                common = [B for B in per_batch['volume'] if B in per_batch['infonce']]
                if common:
                    va = [
                        per_obj['volume'][per_batch['volume'].index(B)] for B in common
                    ]
                    na = [
                        per_obj['infonce'][per_batch['infonce'].index(B)]
                        for B in common
                    ]
                    bstar = crossover_B(common, va, na)
                    pred_star = 2 * (d * k * k + M * (d + T)) / ((k - 1) * d)
                    print(
                        f'\n  crossover (volume == infonce): measured B* = {bstar:.0f}, '
                        f'FLOP model predicts {pred_star:.0f}'
                    )
                    print(
                        f'  ratio at largest common B={common[-1]}: '
                        f'{na[-1] / va[-1]:.2f}x in favour of '
                        f'{"volume" if na[-1] > va[-1] else "infonce"}'
                    )
            print(
                '  slopes fit on the upper half of the sweep; small B is dominated by\n'
                '  kernel-launch overhead and flattens the fit.\n'
            )

    else:
        ks = list(range(2, args.max_k + 1))
        B = args.batch
        print(f'=== B = {B} ===')
        print(
            f'{"k":>3} {"terms(vol)":>11} {"terms(nce)":>11} {"volume ms":>10} {"nce ms":>10}'
        )
        per_obj = {'volume': [], 'infonce': []}
        for k in ks:
            res = {
                o: time_step(o, B, k, d, args, device) for o in ('volume', 'infonce')
            }
            print(
                f'{k:>3} {k:>11} {k * (k - 1) // 2:>11} '
                f'{res["volume"]["ms"]:>10.3f} {res["infonce"]["ms"]:>10.3f}'
            )
            for o in per_obj:
                per_obj[o].append(res[o]['ms'])
                rows.append(
                    {'sweep': 'modalities', 'objective': o, 'B': B, 'k': k, **res[o]}
                )
        for o in per_obj:
            print(f'  {o:>8} log-log slope in k: {loglog_slope(ks, per_obj[o]):.2f}')
        print('  alignment terms: volume k, infonce k(k-1)/2\n')

    if args.out:
        with open(args.out, 'w') as fh:
            json.dump({'args': vars(args), 'rows': rows}, fh, indent=2)
        print(f'wrote {args.out}')
    if args.emit_latex:
        emit_latex(rows, args, args.emit_latex)
    if args.emit_figure:
        emit_figure(rows, args, args.emit_figure)


if __name__ == '__main__':
    main()
