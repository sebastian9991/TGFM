import math
from typing import List

from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class PolynomialDecayLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_updates: int,
        tot_updates: int,
        lr: float,
        end_lr: float,
        power: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float | Tensor]:
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (
                self.tot_updates - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self) -> bool:
        assert False


class WarmupCosineDecayScheduler(_LRScheduler):
    """A PyTorch scheduler combining warmup + cosine decay.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of warmup steps.
        total_steps (int): Total number of steps for the schedule (warmup + cosine).
        max_lr (float): The peak (maximum) learning rate to achieve at the end of warmup.
        last_epoch (int): The index of the last epoch/step. Default: -1 (initial).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        max_lr: float,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float | Tensor]:
        """This method is called by PyTorch each time you do `scheduler.step()`.
        It must return a list of LRs, one per param group.
        """
        step = self.last_epoch  # this gets incremented by 1 on each scheduler.step()

        if step < 0:
            return [0.0 for _ in self.base_lrs]

        if step < self.warmup_steps:
            # Linear warmup from LR=0 up to LR=max_lr
            lr = self.max_lr * (float(step) / float(self.warmup_steps))
        elif step <= self.total_steps:
            # Cosine decay from max_lr down to near 0
            progress = float(step - self.warmup_steps) / float(
                self.total_steps - self.warmup_steps
            )
            lr = 0.5 * self.max_lr * (1.0 + math.cos(math.pi * progress))
        else:
            # If the step exceeds total_steps, you can either clamp or raise an error.
            # We'll raise an error to match your original code:
            raise ValueError(f'Step ({step}) > total_steps ({self.total_steps}).')

        return [lr for _ in self.base_lrs]


class ConstantLRScheduler(_LRScheduler):
    """A scheduler that always returns 'peak_lr' for each param group."""

    def __init__(self, optimizer: Optimizer, peak_lr: float, last_epoch: int = -1):
        self.peak_lr = peak_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float | Tensor]:
        # Return the same LR for all parameter groups.
        return [self.peak_lr for _ in self.base_lrs]
