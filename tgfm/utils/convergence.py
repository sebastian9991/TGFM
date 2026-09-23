"""Convergence monitoring: which objective reaches a given representation
quality first, under one training budget.

Training losses are not comparable across arms (InfoNCE against det G + SIGReg
differ in scale and in their minima), so convergence is read off quantities
computed identically for every arm:

    conv/zeroshot_macro    target-TAG zero-shot accuracy (zeroshot_macro)
    conv/zs_<dataset>      its per-dataset terms
    conv/r1_g2t, r10_g2t,  retrieval over a fixed held-out set of source
    conv/r1_t2g, r10_t2g   pairs, scored with the arm's zeroshot_direction
                           exactly as zero_shot_eval.evaluate scores it
    conv/rankme            effective rank of held-out encode_graph outputs
                           (Garrido et al., ICML 2023); collapse diagnostic

against three clocks, logged under clock/:

    step                   optimizer steps, identical across ranks
    samples                step * batch_size * world_size
    train_s                training wall clock, CUDA-synchronised, paused
                           during evaluation, checkpointing and barriers

Held-out retrieval is the task InfoNCE optimises directly and so favours
GraphCLIP; zero-shot on the targets is the primary curve.

Every eval point writes one parseable line to the log:
    CONV step=... samples=... train_s=... epoch=... zeroshot_macro=... ...
which tools/collect_convergence.py reads.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
import wandb
from torch import Tensor
from torch.nn.functional import normalize
from torch_geometric.loader import DataLoader
from transformers import PreTrainedTokenizerBase

from tgfm.evaluation.zero_shot_eval import zeroshot_macro

HOLDOUT_SEED = 0  # fixed, independent of global_seed: same probe set for every run


def holdout_split(n: int, n_holdout: int) -> Tuple[List[int], List[int]]:
    """(train indices, held-out indices), deterministic in n and n_holdout."""
    if not 0 < n_holdout < n:
        raise ValueError(f'conv_holdout_pairs={n_holdout} must be in (0, {n}).')
    gen = torch.Generator().manual_seed(HOLDOUT_SEED)
    perm = torch.randperm(n, generator=gen)
    hold = perm[:n_holdout].sort().values.tolist()
    train = perm[n_holdout:].sort().values.tolist()
    return train, hold


def eval_schedule(total_steps: int, first_step: int, every: int) -> Set[int]:
    """Step 0, log-spaced steps from first_step, every `every` steps, last step."""
    steps = {0, total_steps}
    k = max(1, first_step)
    while k < total_steps:
        steps.add(k)
        k *= 2
    if every > 0:
        steps.update(range(every, total_steps, every))
    return steps


class TrainClock:
    """Accumulating wall clock with CUDA synchronisation at start and pause."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._total = 0.0
        self._t0: Optional[float] = None

    def _sync(self) -> None:
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def start(self) -> None:
        if self._t0 is None:
            self._sync()
            self._t0 = time.perf_counter()

    def pause(self) -> None:
        if self._t0 is not None:
            self._sync()
            self._total += time.perf_counter() - self._t0
            self._t0 = None

    @property
    def seconds(self) -> float:
        running = time.perf_counter() - self._t0 if self._t0 is not None else 0.0
        return self._total + running


def _uses_text_pred(direction: str) -> bool:
    return direction in ('text_pred', 'text_pred+graph_pred')


def _uses_graph_pred(direction: str) -> bool:
    return direction in ('graph_pred', 'text_pred+graph_pred')


def rankme(z: Tensor, eps: float = 1e-7) -> float:
    """exp of the entropy of the normalised singular values."""
    s = torch.linalg.svdvals(z.float())
    p = s / s.sum() + eps
    return float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def retrieval_probe(
    model: torch.nn.Module,
    graphs: list,
    tokenizer: PreTrainedTokenizerBase,
    max_text_length: int,
    direction: str,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, float]:
    """R@1 / R@10 over the held-out pairs, plus RankMe of encode_graph."""
    was_training = model.training
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    raw_g, zg, zt = [], [], []
    for batch in loader:
        batch_t = tokenizer(
            batch.summary,
            truncation=True,
            padding=True,
            max_length=max_text_length,
            return_tensors='pt',
        ).to(device)
        batch = batch.to(device)
        g = model.encode_graph(batch)
        t = model.encode_text(batch_t['input_ids'], batch_t['attention_mask'])
        raw_g.append(g)
        zg.append(model.graph_predictor(g) if _uses_graph_pred(direction) else g)
        zt.append(model.text_predictor(t) if _uses_text_pred(direction) else t)
    if was_training:
        model.train()

    g_all = normalize(torch.cat(zg).float(), dim=-1)
    t_all = normalize(torch.cat(zt).float(), dim=-1)
    sim = g_all @ t_all.T
    pos = sim.diag()
    # Rank of the true partner = number of candidates scored strictly higher.
    rank_g2t = (sim > pos[:, None]).sum(1)
    rank_t2g = (sim > pos[None, :]).sum(0)
    return {
        'r1_g2t': float((rank_g2t < 1).float().mean()),
        'r10_g2t': float((rank_g2t < 10).float().mean()),
        'r1_t2g': float((rank_t2g < 1).float().mean()),
        'r10_t2g': float((rank_t2g < 10).float().mean()),
        'rankme': rankme(torch.cat(raw_g)),
    }


class ConvMonitor:
    """Runs the eval schedule inside the training loop.

    Evaluation runs on rank 0 on the unwrapped module (no collectives: DDP is
    bypassed and SyncBatchNorm uses running statistics in eval mode); the other
    ranks wait on the gloo barrier. The clock is paused for the whole interval,
    including the barrier.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        model_args: Any,
        data_args: Any,
        holdout_graphs: list,
        schedule: Set[int],
        device: torch.device,
        global_rank: int,
        world_size: int,
        cpu_pg: dist.ProcessGroup,
        verbose: bool,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.data_args = data_args
        self.holdout_graphs = holdout_graphs
        self.schedule = schedule
        self.device = device
        self.global_rank = global_rank
        self.global_batch = model_args.batch_size * world_size
        self.cpu_pg = cpu_pg
        self.verbose = verbose
        self.clock = TrainClock(device)
        self.eval_s = 0.0  # reset by the caller each epoch

        if global_rank == 0:
            logging.info(
                f'Convergence eval at {len(schedule)} points: {sorted(schedule)}'
            )
            if verbose and wandb.run is not None:
                wandb.define_metric('clock/train_s')
                wandb.define_metric('conv/*', step_metric='clock/train_s')

    def maybe_eval(self, step: int, epoch: int) -> None:
        if step not in self.schedule:
            return
        self.clock.pause()
        t0 = time.perf_counter()
        if self.global_rank == 0:
            stats: Dict[str, float] = {}
            if self.model_args.conv_zeroshot:
                macro, per_ds = zeroshot_macro(
                    self.model,
                    self.tokenizer,
                    self.model_args,
                    datasets=self.data_args.target_data.split('+'),
                    seeds=list(range(self.data_args.eval_seeds_sweep)),
                    device=self.device,
                    eval_batch_size=self.data_args.eval_batch_size,
                )
                stats['zeroshot_macro'] = float(macro)
                stats.update({f'zs_{k}': float(v) for k, v in per_ds.items()})
            stats.update(
                retrieval_probe(
                    self.model,
                    self.holdout_graphs,
                    self.tokenizer,
                    self.model_args.max_text_length,
                    self.model_args.zeroshot_direction,
                    self.device,
                    batch_size=self.data_args.eval_batch_size,
                )
            )
            stats['eval_s'] = time.perf_counter() - t0
            clock = {
                'step': float(step),
                'samples': float(step * self.global_batch),
                'train_s': self.clock.seconds,
                'epoch': float(epoch),
            }
            logging.info(
                'CONV ' + ' '.join(f'{k}={v:.6g}' for k, v in (clock | stats).items())
            )
            if self.verbose:
                wandb.log(
                    {f'clock/{k}': v for k, v in clock.items()}
                    | {f'conv/{k}': v for k, v in stats.items()}
                )
        dist.barrier(group=self.cpu_pg)
        self.eval_s += time.perf_counter() - t0
        self.clock.start()
