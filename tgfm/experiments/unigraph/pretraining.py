"""Distributed step-based pretraining for UniGraph.
Launch with:
    torchrun --standalone --nproc_per_node=4 \
        tgfm/experiments/unigraph/pretraining.py \
        --config-file configs/unigraph_pretrain.yaml.
"""

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from tgfm.dataset.mag import MAG240MGraphDataset
from tgfm.dataset.mag_memmap import MAG240MMapTextStore
from tgfm.models.unigraph import UniGraph
from tgfm.utils.args import ModelArguments, UnigraphArguments, parse_args
from tgfm.utils.logger import setup_logging
from tgfm.utils.path import get_root_dir
from tgfm.utils.seed import seed_everything

parser = argparse.ArgumentParser(
    description='Distributed step-based pretraining UniGraph.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file', type=str, required=True, help='Path to configuration file.'
)


def setup_distributed() -> Tuple[int, int, int, torch.device]:
    """Set up distributed backend, get ranks, world size."""
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0
    assert torch.distributed.is_available()

    if 'LOCAL_RANK' not in os.environ:
        raise RuntimeError('Launch with torchrun (sets LOCAL_RANK).')

    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    return local_rank, global_rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def infinite_loader(loader: NeighborLoader) -> Iterator[Data]:
    """Yield batches forever. When the loader exhausts, start a new pass."""
    while True:
        for batch in loader:
            yield batch


def save_checkpoint(
    path: Path,
    step: int,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    ema_loss: float,
    best_ema_loss: float,
    world_size: int,
) -> None:
    """Save a checkpoint. Called only on rank 0.

    Uses tmp-file + rename for atomicity: if the job dies mid-save, the old
    checkpoint is still valid and the partial .tmp file is just debris.
    """
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(
        {
            'step': step,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ema_loss': ema_loss,
            'best_ema_loss': best_ema_loss,
            'world_size': world_size,
            'rng_states': {
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state(),
                'numpy': np.random.get_state(),
                'python': random.getstate(),
                'torch_cpu': torch.random.get_rng_state(),
                'torch_gpu': torch.cuda.get_rng_state_all(),
            },
        },
        tmp_path,
    )
    tmp_path.rename(path)


def load_checkpoint(
    path: Path,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Load a checkpoint. Returns metadata dict. Called on all ranks."""
    # TODO: Look at how load_checkpoint is implemented in the mila docs. In the multi-node/multi-gpu example.
    ckpt = torch.load(path, map_location=device)
    model.module.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'rng_states' in ckpt:
        torch.set_rng_state(ckpt['rng_states']['torch'])
        torch.cuda.set_rng_state(ckpt['rng_states']['cuda'])
        np.random.set_state(ckpt['rng_states']['numpy'])
        random.setstate(ckpt['rng_states']['python'])
    return {
        'step': ckpt['step'],
        'ema_loss': ckpt['ema_loss'],
        'best_ema_loss': ckpt['best_ema_loss'],
    }


def train_steps(
    model: DDP,
    train_loader: NeighborLoader,
    text_store: MAG240MMapTextStore,
    optimizer: torch.optim.AdamW,
    device: torch.device,
    global_rank: int,
    world_size: int,
    save_dir: Path,
    max_steps: int,
    save_every_steps: int,
    log_every_steps: int,
    reduce_every_steps: int,
    ema_alpha: float,
    max_wall_seconds: Optional[float],
    start_step: int = 0,
    best_ema_loss_init: float = float('inf'),
    verbose: bool = False,
) -> None:
    """Run step-based training with best-checkpoint tracking."""
    model.train()

    data_iter = infinite_loader(train_loader)
    wall_start = time.perf_counter()

    ema_loss_local: Optional[float] = None
    ema_loss_global: float = float('inf')
    best_ema_loss = best_ema_loss_init

    pbar = tqdm(
        total=max_steps,
        initial=start_step,
        desc='Training',
        disable=(global_rank != 0),
        smoothing=0.1,
    )

    step = start_step
    for step in range(start_step, max_steps):
        batch = next(data_iter)
        optimizer.zero_grad(set_to_none=True)

        text_features = text_store.get_features(batch.n_id, apply_masking=True)
        input_ids = text_features['input_ids'].to(device, non_blocking=True)
        masked_input_ids = text_features['masked_input_ids'].to(
            device, non_blocking=True
        )
        attention_mask = text_features['attention_mask'].to(device, non_blocking=True)
        token_type_ids = text_features['token_type_ids'].to(device, non_blocking=True)
        edge_index = batch.edge_index.to(device, non_blocking=True)

        loss, latent_loss = model(
            input_ids,
            masked_input_ids,
            attention_mask,
            token_type_ids,
            edge_index,
            device,
        )

        loss.backward()
        optimizer.step()
        model.module._update_target_networks()

        # Update local EMA loss (cheap, no sync needed)
        loss_val = loss.detach().item()
        if ema_loss_local is None:
            ema_loss_local = loss_val
        else:
            ema_loss_local = ema_alpha * ema_loss_local + (1 - ema_alpha) * loss_val

        # Periodic global sync for "best" comparison and accurate logging.
        # An all-reduce every step would waste bandwidth; every ~500 steps
        # is frequent enough to catch improvements without killing throughput.
        if (step + 1) % reduce_every_steps == 0:
            loss_tensor = torch.tensor(ema_loss_local, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            ema_loss_global = loss_tensor.item()

            if ema_loss_global < best_ema_loss:
                best_ema_loss = ema_loss_global
                if global_rank == 0:
                    save_checkpoint(
                        save_dir / 'best.pt',
                        step + 1,
                        model,
                        optimizer,
                        ema_loss_global,
                        best_ema_loss,
                        world_size,
                    )
                    logging.info(
                        f'[step {step + 1}] New best loss {ema_loss_global:.4f} — saved best.pt'
                    )

        # Logging
        if global_rank == 0 and (step + 1) % log_every_steps == 0:
            elapsed = time.perf_counter() - wall_start
            steps_done = step + 1 - start_step
            rate = steps_done / elapsed
            eta_s = (max_steps - step - 1) / max(rate, 1e-9)
            pbar.set_postfix(
                {
                    'loss': f'{loss_val:.4f}',
                    'ema': f'{ema_loss_global:.4f}',
                    'best': f'{best_ema_loss:.4f}',
                    'rate': f'{rate:.2f}it/s',
                    'eta_h': f'{eta_s / 3600:.1f}',
                }
            )
            logging.info(
                f'[step {step + 1}/{max_steps}] loss={loss_val:.4f} '
                f'ema={ema_loss_global:.4f} best={best_ema_loss:.4f} '
                f'rate={rate:.2f}it/s eta={eta_s / 3600:.1f}h'
            )

            if verbose:
                wandb.log(
                    {
                        'train/step': step,
                        'train/loss': loss_val,
                        'train/ema_loss_local': ema_loss_local,
                        'train/ema_loss_global': ema_loss_global,
                    }
                )

        # Periodic "latest" checkpoint for resumption
        if global_rank == 0 and (step + 1) % save_every_steps == 0:
            save_checkpoint(
                save_dir / 'latest.pt',
                step + 1,
                model,
                optimizer,
                ema_loss_global,
                best_ema_loss,
                world_size,
            )

        pbar.update(1)

        # Wall-clock stop (graceful SLURM exit).
        # All ranks check and all-reduce so they all exit together — otherwise
        # one rank exiting early would leave others hanging on NCCL sync.
        if max_wall_seconds is not None:
            elapsed = time.perf_counter() - wall_start
            should_stop = torch.tensor(
                1 if elapsed > max_wall_seconds else 0, device=device
            )
            dist.all_reduce(should_stop, op=dist.ReduceOp.SUM)
            if should_stop.item() > 0:
                if global_rank == 0:
                    logging.warning(
                        f'Wall clock limit reached at step {step + 1}. '
                        f'Saving final checkpoint and exiting.'
                    )
                    save_checkpoint(
                        save_dir / 'latest.pt',
                        step + 1,
                        model,
                        optimizer,
                        ema_loss_global,
                        best_ema_loss,
                        world_size,
                    )
                break

    pbar.close()

    if global_rank == 0:
        save_checkpoint(
            save_dir / 'latest.pt',
            step + 1,
            model,
            optimizer,
            ema_loss_global,
            best_ema_loss,
            world_size,
        )
        logging.info(
            f'Training complete at step {step + 1}. '
            f'Final ema_loss={ema_loss_global:.4f}, best_ema_loss={best_ema_loss:.4f}'
        )
        if verbose:
            wandb.finish()


def run_unigraph(
    model_args: ModelArguments,
    dataset: MAG240MGraphDataset,
    text_store: MAG240MMapTextStore,
    save_dir: Path,
    local_rank: int,
    global_rank: int,
    world_size: int,
    device: torch.device,
    full_coverage: bool = False,
    verbose: bool = False,
) -> None:
    assert isinstance(model_args, UnigraphArguments)
    data = dataset[0]

    if global_rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f'Graph: {data.num_nodes} nodes, {data.num_edges} edges')
        logging.info(f'World size: {world_size}')
        logging.info(
            f'Effective (global) batch size: {model_args.batch_size * world_size}'
        )

    # Shard seed nodes across ranks
    all_nodes = torch.arange(data.num_nodes)
    nodes_this_rank = all_nodes.tensor_split(world_size)[global_rank]

    loader = NeighborLoader(
        data,
        input_nodes=nodes_this_rank,
        num_neighbors=model_args.num_neighbors,
        batch_size=model_args.batch_size,
        shuffle=True,
        num_workers=5,
        prefetch_factor=2,
        persistent_workers=True,
    )

    model = UniGraph(model_args).to(device)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_args.lr,
        weight_decay=model_args.weight_decay,
    )

    start_step = 0
    best_ema_loss_init = float('inf')
    latest_path = save_dir / 'latest.pt'
    if latest_path.exists():
        if global_rank == 0:
            logging.info(f'Resuming from {latest_path}')
            logging.info(
                f'NOTE: data iterator restarts from the beginning of the shuffle order. A certain amount of early seed nodes may be revisted.'
            )
        meta = load_checkpoint(latest_path, model, optimizer, device)
        start_step = meta['step']
        best_ema_loss_init = meta['best_ema_loss']
        if global_rank == 0:
            logging.info(
                f'Resumed at step {start_step}, best_ema_loss={best_ema_loss_init:.4f}'
            )

    max_steps = getattr(model_args, 'max_steps', 500_000)
    if full_coverage:
        total_nodes = data.num_nodes
        nodes_per_step = model_args.batch_size * world_size
        steps_per_epoch = total_nodes // nodes_per_step
        max_steps = max(max_steps, steps_per_epoch)
    save_every_steps = getattr(model_args, 'save_every_steps', 5_000)
    log_every_steps = getattr(model_args, 'log_every_steps', 100)
    reduce_every_steps = getattr(model_args, 'reduce_every_steps', 500)
    ema_alpha = getattr(model_args, 'loss_ema_alpha', 0.98)
    max_wall_seconds = getattr(model_args, 'max_wall_seconds', None)

    if global_rank == 0:
        logging.info(
            f'Training config: max_steps={max_steps}, '
            f'save_every={save_every_steps}, '
            f'log_every={log_every_steps}, '
            f'reduce_every={reduce_every_steps}, '
            f'ema_alpha={ema_alpha}, '
            f'max_wall_seconds={max_wall_seconds}'
        )
        if verbose:
            wandb.config.update(
                {
                    'lr': model_args.lr,
                    'batch_size': model_args.batch_size,
                    'num_neighbors': model_args.num_neighbors,
                    'max_steps': max_steps,
                    'reduce_every_steps': reduce_every_steps,
                    'ema_alpha': ema_alpha,
                }
            )

    train_steps(
        model=model,
        train_loader=loader,
        text_store=text_store,
        optimizer=optimizer,
        device=device,
        global_rank=global_rank,
        world_size=world_size,
        save_dir=save_dir,
        max_steps=max_steps,
        save_every_steps=save_every_steps,
        log_every_steps=log_every_steps,
        reduce_every_steps=reduce_every_steps,
        ema_alpha=ema_alpha,
        max_wall_seconds=max_wall_seconds,
        start_step=start_step,
        best_ema_loss_init=best_ema_loss_init,
        verbose=verbose,
    )


def main() -> None:
    local_rank, global_rank, world_size, device = setup_distributed()

    try:
        root = get_root_dir()
        args = parser.parse_args()
        config_file_path = root / args.config_file
        meta_args, experiment_args = parse_args(config_file_path)
        if meta_args.verbose and global_rank == 0:
            mode = 'online' if getattr(meta_args, 'wandb_online', True) else 'offline'
            wandb.init(
                project=getattr(meta_args, 'wandb_project', 'unigraph-pretrain'),
                name=getattr(meta_args, 'wandb_run_name', None),
                config={
                    'world_size': world_size,
                    'global_seed': meta_args.global_seed,
                },
                mode=mode,
            )
            time.sleep(5)  # Helpful for a pottential 409 error on wandb servers.

        seed_everything(meta_args.global_seed + global_rank)

        if global_rank == 0:
            setup_logging(meta_args.log_file_path)
        else:
            logging.basicConfig(level=logging.WARNING)

        root_dir = Path(str(meta_args.root_dir))
        dataset = MAG240MGraphDataset(root=str(root_dir))
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        text_store = MAG240MMapTextStore(
            output_dir=str(root_dir / 'mag240m_mapping'),
            tokenizer=tokenizer,
        )

        if global_rank == 0:
            logging.info('Dataset, tokenizer, and text store loaded.')
            logging.info(
                f'World size: {world_size}, global rank: {global_rank}, "local_rank: {local_rank}'
            )

        for experiment, experiment_arg in experiment_args.exp_args.items():
            if global_rank == 0:
                logging.info(f'\n***Running*** {experiment}')
            run_unigraph(
                model_args=experiment_arg.model_args,
                dataset=dataset,
                text_store=text_store,
                save_dir=root_dir / 'weights',
                local_rank=local_rank,
                global_rank=global_rank,
                world_size=world_size,
                device=device,
                verbose=meta_args.verbose,
            )
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
