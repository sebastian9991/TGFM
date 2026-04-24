"""Distributed pretraining for UniGraph.

Launch with:
    torchrun --standalone --nproc_per_node=8 \
        tgfm/experiments/unigraph/pretraining.py \
        --config-file configs/unigraph_pretrain.yaml

For multi-node (e.g. SLURM), replace --standalone with:
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
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
    description='Distributed pre-training UniGraph on ogb-100M.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file', type=str, required=True, help='Path to configuration file.'
)


def setup_distributed() -> Tuple[int, int, int, torch.device]:
    """Initialize the process group. Returns (local_rank, global_rank, world_size, device).

    torchrun sets LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.
    """
    if 'LOCAL_RANK' not in os.environ:
        raise RuntimeError(
            'This script must be launched with torchrun. Example:\n'
            '  torchrun --standalone --nproc_per_node=8 pretraining.py ...'
        )

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


def is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def train_pretrain(
    model: nn.Module,
    train_loader: DataLoader,
    text_store: MAG240MMapTextStore,
    optimizer: torch.optim.AdamW,
    epoch: int,
    device: torch.device,
    global_rank: int,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_latent_loss = 0.0
    num_steps = 0

    # Only rank 0 shows the progress bar — prevents 8× duplicated tqdm lines.
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', disable=(global_rank != 0))

    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)

        text_features = text_store.get_features(batch.n_id, apply_masking=True)

        # non_blocking transfers overlap with prior GPU work
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

        # EMA update on the unwrapped model. Runs independently on each rank —
        # since online params are synced by DDP, the EMA produces identical
        # target params on every rank (no extra all-reduce needed).
        if model.module.args.lam > 0:
            model.module._update_target_networks(
                tau=getattr(model.module.args, 'ema_tau', 0.996)
            )

        # Accumulate loss without forcing sync every step.
        # .detach() keeps it on GPU; we .item() only periodically for display.
        total_loss += loss.detach().item()
        total_latent_loss += latent_loss.detach().item()
        num_steps += 1

        if global_rank == 0 and num_steps % 10 == 0:
            pbar.set_postfix(
                {
                    'loss': f'{loss.item():.4f}',
                    'latent': f'{latent_loss.item():.4f}',
                }
            )

    # Average losses across ranks for a global epoch-level metric.
    # Each rank has its own mean; we reduce to get the true global mean.
    avg_loss = torch.tensor(total_loss / max(num_steps, 1), device=device)
    avg_latent = torch.tensor(total_latent_loss / max(num_steps, 1), device=device)
    dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
    dist.all_reduce(avg_latent, op=dist.ReduceOp.AVG)

    return avg_loss.item(), avg_latent.item()


def run_unigraph(
    model_args: ModelArguments,
    dataset: MAG240MGraphDataset,
    text_store: MAG240MMapTextStore,
    save_dir: Path,
    local_rank: int,
    global_rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    assert isinstance(model_args, UnigraphArguments)
    data = dataset[0]

    if global_rank == 0:
        logging.info(f'Graph: {data.num_nodes} nodes, {data.num_edges} edges')
        logging.info(f'World size: {world_size}')

    # ---- Shard seed nodes across ranks ----
    # Each rank samples from a disjoint slice of the graph. No rank sees
    # another rank's seed nodes, so we cover all nodes exactly once per epoch.
    all_nodes = torch.arange(data.num_nodes)
    # tensor_split handles uneven divisions gracefully
    nodes_this_rank = all_nodes.tensor_split(world_size)[global_rank]

    if global_rank == 0:
        logging.info(
            f'Rank 0 sees {len(nodes_this_rank)} seed nodes '
            f'(of {data.num_nodes} total). Each rank has ~{data.num_nodes // world_size}.'
        )

    loader = NeighborLoader(
        data,
        input_nodes=nodes_this_rank,
        num_neighbors=model_args.num_neighbors,
        batch_size=model_args.batch_size,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # ---- Build and wrap the model ----
    model = UniGraph(model_args).to(device)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        # EMA updates mutate target parameters which never see gradients.
        # DDP by default warns about unused parameters; we need to suppress.
        find_unused_parameters=False,
        # Small memory saving — stores grads as views over bucket tensors
        gradient_as_bucket_view=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_args.lr,
        weight_decay=model_args.weight_decay,
    )

    for epoch in range(model_args.epochs):
        # Barrier so all ranks start the epoch together. Helps with logging sanity.
        dist.barrier()

        epoch_start = time.perf_counter()
        pretrain_loss, pretrain_latent_loss = train_pretrain(
            model,
            loader,
            text_store,
            optimizer,
            epoch,
            device,
            global_rank,
        )
        epoch_time = time.perf_counter() - epoch_start

        if global_rank == 0:
            logging.info(
                f'[Epoch {epoch}] loss={pretrain_loss:.4f} '
                f'latent={pretrain_latent_loss:.4f} time={epoch_time:.1f}s'
            )

            # Only rank 0 writes checkpoints. All ranks have identical weights
            # post-optimizer.step(), so this is safe.
            # Removing condition
            ckpt_path = save_dir / f'pretrain_epoch_{epoch + 1}.pt'
            torch.save(
                {
                    'epoch': epoch,
                    # Save .module to unwrap DDP — checkpoint is loadable
                    # without DDP later (e.g., for inference).
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'pretrain_loss': pretrain_loss,
                    'pretrain_latent_loss': pretrain_latent_loss,
                    'world_size': world_size,  # for sanity-check on reload
                },
                ckpt_path,
            )
            logging.info(f'Saved checkpoint to {ckpt_path}')

        # All ranks wait for rank 0 to finish saving before proceeding.
        dist.barrier()


def main() -> None:
    local_rank, global_rank, world_size, device = setup_distributed()

    try:
        root = get_root_dir()
        args = parser.parse_args()
        config_file_path = root / args.config_file
        meta_args, experiment_args = parse_args(config_file_path)

        # Seed each rank differently so NeighborLoader's shuffle diverges.
        # Without this, every rank would sample the same random orderings.
        seed_everything(meta_args.global_seed + global_rank)

        if global_rank == 0:
            setup_logging(meta_args.log_file_path)
        else:
            logging.basicConfig(level=logging.WARNING)

        root_dir = Path(str(meta_args.root_dir))
        dataset = MAG240MGraphDataset(root=str(root_dir))

        # Every rank opens its own tokenizer and text store. The memmap is
        # shared via OS page cache — no duplicate RAM usage for the data itself.
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        text_store = MAG240MMapTextStore(
            output_dir=str(root_dir / 'mag240m_mapping'),
            tokenizer=tokenizer,
        )

        if global_rank == 0:
            logging.info('Dataset, tokenizer, and text store loaded on all ranks.')

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
            )
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
