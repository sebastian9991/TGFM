"""End-to-end step and epoch timing for the pretraining loop.

Reports what the complexity appendix does not: whether the asymptotic
separation between the alignment terms is visible in a real training step, at
the batch sizes we actually run. It usually is not at small B, because the
encoder dominates, and saying so with a measurement is stronger than leaving a
reviewer to ask.

What is measured per epoch:

    epoch_s          wall-clock seconds, synchronised at both ends
    step_ms          per-step total, median and mean over sampled steps
    forward_ms       encoder forward
    loss_ms          alignment term forward ONLY (see the caveat below)
    backward_ms      backward through loss and encoder together
    optim_ms         optimizer and scheduler step
    data_ms          everything before the forward: tokenisation, host-to-
                     device transfer, augmentation
    throughput       subgraphs per second, summed across ranks
    peak_mem_mib     peak allocated memory, max across ranks

Caveat on attribution. `.backward()` traverses the alignment term and the
encoder in a single call, so `loss_ms` is the alignment FORWARD only; its
backward is inside `backward_ms` and cannot be separated without a second
pass. For the alignment term's isolated forward+backward cost, use
tgfm/tools/bench_alignment_cost.py, which feeds both objectives synthetic
embeddings. The two measurements answer different questions and the paper
needs both: this one for "does it matter end to end", that one for "is the
term linear or quadratic".

Sampling. CUDA events are asynchronous, but reading an elapsed time forces a
host synchronisation, which perturbs the thing being measured. Only one step
in `sample_every` is instrumented, and the first `warmup` steps are skipped so
that allocator warmup and cuDNN autotuning do not enter the statistics.

Distributed. Under DDP the all-reduce in the backward pass makes every rank
wait for the slowest, so per-step times are reduced with MAX rather than
averaged; item counts are summed.
"""

import logging
import statistics
import time
from collections import defaultdict
from typing import Dict, List

import torch
import torch.distributed as dist


class StepTimer:
    """Sampled per-step timing with an exact epoch wall clock.

    Usage inside the training loop::

        timer.epoch_begin()
        for batch in loader:
            timer.step_begin()
            ...                      # tokenise, move to device, augment
            timer.mark('data')
            out = model(batch, batch_t)
            timer.mark('forward')
            losses = criterion(out)
            timer.mark('loss')
            losses['loss'].backward()
            timer.mark('backward')
            optimizer.step(); scheduler.step()
            timer.mark('optim')
            timer.step_end(n_items=batch.num_graphs)
        stats = timer.epoch_end()
    """

    REGIONS = ('data', 'forward', 'loss', 'backward', 'optim')

    def __init__(
        self,
        device: torch.device,
        world_size: int = 1,
        sample_every: int = 50,
        warmup: int = 10,
        enabled: bool = True,
    ) -> None:
        self.device = device
        self.world_size = world_size
        self.sample_every = max(1, sample_every)
        self.warmup = warmup
        self.enabled = enabled and device.type == 'cuda'
        self.planned_samples = 0
        self._acc: Dict[str, List[float]] = defaultdict(list)
        self._events: List = []
        self._sampled = False
        self._n_steps = 0
        self._items = 0
        self._t0 = 0.0

    def plan(self, steps_per_epoch: int, target_samples: int = 24) -> None:
        """Choose sample_every and warmup from the epoch's actual step count.

        A batch-size sweep changes the number of steps in an epoch: at the same
        pool, doubling B halves the steps. With a fixed period the large-B
        epochs end up with one or two sampled steps, or with none at all when
        the epoch is shorter than `warmup`, and the median is then taken over
        nothing and reported as 0.0.

        Two properties matter for the comparison and both follow from fixing
        the sample COUNT rather than the period. Every batch size contributes
        the same number of medians, so no point on the curve is noisier than
        the others. And reading a CUDA event costs one host synchronisation
        per sampled step, so a fixed count keeps that cost constant instead of
        letting it grow as the epochs get longer, which would otherwise show up
        as an apparent per-step cost that has nothing to do with the objective.

        Call after the loader exists and before the first epoch.
        """
        steps = int(steps_per_epoch)
        if steps <= 0:
            return
        # Keep the requested warmup when the epoch can afford it; a short epoch
        # gets a fifth of its steps, so there is always something left to
        # sample.
        self.warmup = min(self.warmup, max(1, steps // 5))
        usable = max(1, steps - self.warmup)
        self.sample_every = max(1, usable // max(1, target_samples))
        self.planned_samples = usable // self.sample_every
        logging.info(
            f'TIMING-PLAN steps={steps} warmup={self.warmup} '
            f'sample_every={self.sample_every} expected_samples={self.planned_samples}'
        )

    def epoch_begin(self) -> None:
        self._acc.clear()
        self._n_steps = 0
        self._items = 0
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._t0 = time.perf_counter()

    def step_begin(self) -> bool:
        """Start a step; returns whether this one is instrumented."""
        self._n_steps += 1
        self._sampled = (
            self.enabled
            and self._n_steps > self.warmup
            and self._n_steps % self.sample_every == 0
        )
        if self._sampled:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._events = [('start', ev)]
        return self._sampled

    def mark(self, name: str) -> None:
        """Close the region that ends here."""
        if self._sampled:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._events.append((name, ev))

    def step_end(self, n_items: int) -> None:
        self._items += int(n_items)
        if not self._sampled:
            return
        # Reading elapsed_time requires the events to have completed.
        torch.cuda.synchronize(self.device)
        prev = self._events[0][1]
        for name, ev in self._events[1:]:
            self._acc[name].append(prev.elapsed_time(ev))
            prev = ev
        self._acc['step'].append(self._events[0][1].elapsed_time(self._events[-1][1]))
        self._events = []
        self._sampled = False

    def _max_keys(self) -> List[str]:
        """Fixed key schema for the collective, in a fixed order.

        tensor_split gives ranks unequal shard sizes, so two ranks can differ
        in step count and therefore in which regions ever got sampled. Building
        the reduce tensor from whatever each rank happens to have populated
        would produce mismatched shapes and hang the all_reduce. The schema is
        fixed instead and unsampled entries are sent as 0.0, which MAX then
        discards in favour of a rank that did sample.
        """
        keys = ['epoch_s', 'steps', 'peak_mem_mib']
        for name in ('step',) + self.REGIONS:
            keys += [f'{name}_ms', f'{name}_ms_mean']
        return keys

    def epoch_end(self) -> Dict[str, float]:
        """Reduce across ranks and return the epoch's timing summary."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

        out: Dict[str, float] = {
            'epoch_s': time.perf_counter() - self._t0,
            'steps': float(self._n_steps),
            'peak_mem_mib': (
                torch.cuda.max_memory_allocated(self.device) / 2**20
                if self.device.type == 'cuda'
                else 0.0
            ),
        }
        for name in ('step',) + self.REGIONS:
            vals = self._acc.get(name, [])
            out[f'{name}_ms'] = statistics.median(vals) if vals else 0.0
            out[f'{name}_ms_mean'] = statistics.fmean(vals) if vals else 0.0
        items = float(self._items)

        if dist.is_available() and dist.is_initialized():
            # Slowest rank gates every step under DDP, so take MAX for times
            # and peak memory; items are summed to give global throughput.
            keys = self._max_keys()
            t = torch.tensor([out[k] for k in keys], device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            out.update(dict(zip(keys, t.tolist())))
            n = torch.tensor([items], device=self.device)
            dist.all_reduce(n, op=dist.ReduceOp.SUM)
            items = float(n.item())

        n_sampled = len(self._acc.get('step', []))
        out['items'] = items
        out['sampled_steps'] = float(n_sampled)
        out['throughput_per_s'] = items / max(out['epoch_s'], 1e-9)
        if out['step_ms'] > 0:
            # Alignment FORWARD as a share of the step. Its backward is inside
            # backward_ms and is not separable here; see the module docstring.
            out['loss_fwd_share'] = out['loss_ms'] / out['step_ms']
        else:
            out['loss_fwd_share'] = 0.0
        return out

    @staticmethod
    def format(stats: Dict[str, float]) -> str:
        def g(k: str, spec: str = '.1f') -> str:
            v = stats.get(k)
            return format(v, spec) if isinstance(v, float) else '-'

        return (
            f'epoch={g("epoch_s")}s steps={int(stats.get("steps", 0))} '
            f'step={g("step_ms", ".2f")}ms '
            f'[data={g("data_ms", ".2f")} fwd={g("forward_ms", ".2f")} '
            f'loss={g("loss_ms", ".2f")} bwd={g("backward_ms", ".2f")} '
            f'opt={g("optim_ms", ".2f")}] '
            f'loss_fwd_share={g("loss_fwd_share", ".4f")} '
            f'thru={g("throughput_per_s", ".0f")}/s '
            f'peak={g("peak_mem_mib")}MiB '
            f'samples={int(stats.get("sampled_steps", 0))}'
        )


def config_fingerprint(
    model_args, world_size: int, n_trainable: int
) -> Dict[str, object]:
    """The knobs that must match before two timing runs are comparable.

    A timing comparison across objectives is only meaningful when everything
    here is equal except `align_objective`. Logging it makes a confound
    visible in the run record rather than leaving it to be discovered later --
    the 6-layer versus 12-layer mismatch between our own earlier runs is
    exactly the case this is meant to catch.
    """
    fields = (
        'align_objective',
        'batch_size',
        'graph_num_layers',
        'graph_hidden_dim',
        'embed_dim',
        'predictor_depth',
        'predictor_hidden_dim',
        'proj_hidden_dim',
        'sigreg_num_slices',
        'sigreg_num_quad_points',
        'graph_aug',
        'adversarial',
        'adv_steps',
        'use_text',
        'use_image',
        'max_text_length',
    )
    fp: Dict[str, object] = {
        f: getattr(model_args, f) for f in fields if hasattr(model_args, f)
    }
    fp['world_size'] = world_size
    fp['global_batch'] = getattr(model_args, 'batch_size', 0) * world_size
    fp['trainable_params'] = n_trainable
    return fp


def log_fingerprint(fp: Dict[str, object]) -> None:
    logging.info('TIMING-CONFIG: ' + ' '.join(f'{k}={v}' for k, v in fp.items()))
