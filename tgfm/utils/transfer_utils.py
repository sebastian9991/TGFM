"""Utilities trimmed from GSTBench's utils.py to the functions used by
train_ssl.py and eval_helper.py for the linear probing experiment.
"""

import random
import time

import numpy as np
import torch
import torch.distributed as dist


def init_process_group(world_size: int, rank: int, port: int = 12345) -> None:
    dist.init_process_group(
        backend='nccl',  # change to 'nccl' for multiple GPUs
        init_method=f'tcp://127.0.0.1:{port}',
        world_size=world_size,
        rank=rank,
    )


def estimate_remaining_time(
    start_time: float, current_batch: int, total_batches: int, k: int
) -> None:
    """Estimates and prints the remaining training time every K batches.

    :param start_time: The time when the epoch started.
    :param current_batch: The current batch number.
    :param total_batches: The total number of batches in the epoch.
    :param k: The function will print the estimated time every K batches.
    """
    if current_batch % k == 0 and current_batch > 0:
        elapsed_time = time.time() - start_time
        batches_processed = current_batch
        avg_time_per_batch = elapsed_time / batches_processed
        remaining_batches = total_batches - current_batch
        estimated_time = avg_time_per_batch * remaining_batches

        # Convert estimated time to minutes and seconds for better readability
        estimated_minutes = int(estimated_time // 60)
        estimated_seconds = int(estimated_time % 60)

        print(f'Estimated Time Remaining: {estimated_minutes}m {estimated_seconds}s')


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
