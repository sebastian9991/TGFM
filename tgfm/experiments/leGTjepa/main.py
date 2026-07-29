"""Distributed pretraining for LeGTJEPA.
Launch with:
    torchrun --standalone --nproc_per_node=4 \
        tgfm/experiments/legtjepa/main.py \
        --config-file configs/legtjepa.yaml.

Expects GraphCLIP's released data layout under ``model_args.data_root``:
    processed_data/{name}.pt      graph structure + SBERT node features
    summary/summary-{name}.json   subgraph-summary pairs

``parse_source_data`` is imported from the GraphCLIP repo (utils/process.py).
"""

import argparse
import logging
import math
import os
import time
from pathlib import Path
from typing import Literal, Tuple

import torch
import torch.distributed as dist
import wandb
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tgfm.evaluation.zero_shot_eval import zeroshot_macro
from tgfm.models.legtjepa import LeGTJEPA
from tgfm.models.losses.legtjepaloss import LeGTJEPALoss
from tgfm.utils.args import (
    DataArguments,
    LeGTJEPAArguments,
    MetaArguments,
    ModelArguments,
    parse_args,
)
from tgfm.utils.logger import setup_logging
from tgfm.utils.path import get_root_dir, get_scratch
from tgfm.utils.process import parse_source_data  # from the GraphCLIP repo
from tgfm.utils.seed import seed_everything
from tgfm.views.augmentations import batch_graph_aug

parser = argparse.ArgumentParser(
    description='Distributed pretraining LeGTJEPA.',
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


def warmup_cosine(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> LambdaLR:
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, fn)


def load_source_graphs(meta_args: MetaArguments, model_args: ModelArguments) -> list:
    assert isinstance(model_args, LeGTJEPAArguments)
    scratch = get_scratch()
    path = scratch / str(meta_args.root_dir) / 'processed'
    graphs: list = []
    for name in model_args.source_data.split('+'):
        data = torch.load(path / f'{name}.pt', weights_only=False)
        graphs.extend(parse_source_data(name, data))
        logging.info(
            f'Loaded source dataset {name} (running total: {len(graphs)} subgraphs)'
        )
    return graphs


def train_epoch(
    model: DDP,
    criterion: LeGTJEPALoss,
    loader: DataLoader,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.AdamW,
    scheduler: LambdaLR,
    model_args: ModelArguments,
    data_args: DataArguments,
    device: torch.device,
    global_rank: int,
    epoch: int,
    log_every_steps: int,
    max_text_length: int,
    verbose: bool,
) -> float:
    """One pass over the rank-local shard; returns the epoch's mean loss."""
    assert isinstance(model_args, LeGTJEPAArguments)
    model.train()
    total_loss, num_batches = 0.0, 0

    pbar = tqdm(
        loader,
        total=len(loader),
        desc=f'Epoch {epoch}',
        disable=(global_rank != 0),
        smoothing=0.1,
        leave=False,
    )

    for step, batch in enumerate(pbar):
        optimizer.zero_grad(set_to_none=True)

        batch_t = tokenizer(
            batch.summary,
            truncation=True,
            padding=True,
            max_length=max_text_length,
            return_tensors='pt',
        )
        batch = batch.to(device, non_blocking=True)
        batch_t = {k: v.to(device, non_blocking=True) for k, v in batch_t.items()}

        if model_args.graph_aug:
            batch = batch_graph_aug(
                batch, model_args.aug_feat_drop, model_args.aug_edge_drop
            )

        if not model_args.adversarial:
            losses = criterion(model(batch, batch_t))
            losses['loss'].backward()
        else:
            m, eps = model_args.adv_steps, model_args.adv_step_size
            x_clean = batch.x
            perturb = torch.empty_like(x_clean).uniform_(-eps, eps).requires_grad_()
            for i in range(m):
                batch.x = x_clean + perturb
                losses = criterion(model(batch, batch_t))
                (losses['loss'] / m).backward()
                if i < m - 1:
                    with torch.no_grad():
                        assert perturb.grad is not None
                        perturb += eps * perturb.grad.sign()
                        perturb.grad.zero_()
            batch.x = x_clean

        optimizer.step()
        scheduler.step()

        total_loss += losses['loss'].detach().item()
        num_batches += 1

        if (
            global_rank == 0
            and epoch % data_args.eval_every_epochs == 0
            or epoch == model_args.epochs
        ):
            if verbose:
                logging.info('Evaluation Model.')
                macro, per_ds = zeroshot_macro(
                    model.module,
                    tokenizer,
                    model_args,
                    datasets=data_args.target_data.split('+'),
                    seeds=list(range(data_args.eval_seeds_sweep)),
                    device=device,
                    eval_batch_size=data_args.eval_batch_size,
                )
                logging.info(f'[epoch {epoch}] zeroshot_macro={100 * macro:2f}')
                logging.info(f'{ {k: f"{100 * v::.2f}" for k, v in per_ds.items()} }')

                wandb.log(
                    {
                        'eval/zeroshot_macro': macro,
                        **{f'eval/zeroshot_{k}': v for k, v in per_ds.items()},
                    }
                )
            dist.barrier()

        if global_rank == 0 and (step + 1) % log_every_steps == 0:
            pbar.set_postfix(
                {
                    'loss': f'{losses["loss"].item():.4f}',
                    'cross': f'{losses["cross"].item():.4f}',
                    'sg_g': f'{losses["sigreg_graph"].item():.4f}',
                    'sg_t': f'{losses["sigreg_text"].item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                }
            )
            logging.info(
                f'[epoch {epoch} step {step + 1}/{len(loader)}] '
                f'loss={losses["loss"].item():.4f} '
                f'cross={losses["cross"].item():.4f} '
                f'sigreg_g={losses["sigreg_graph"].item():.4f} '
                f'sigreg_t={losses["sigreg_text"].item():.4f} '
                f'lr={scheduler.get_last_lr()[0]:.2e}'
            )
            if verbose:
                wandb.log(
                    {
                        'train/epoch': epoch,
                        'train/loss': losses['loss'].item(),
                        'train/cross': losses['cross'].item(),
                        'train/sigreg_graph': losses['sigreg_graph'].item(),
                        'train/sigreg_text': losses['sigreg_text'].item(),
                        'train/lr': scheduler.get_last_lr()[0],
                    }
                )

    # Average across ranks so the logged epoch loss reflects the global shard.
    loss_tensor = torch.tensor(total_loss / max(1, num_batches), device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    return loss_tensor.item()


def run_legtjepa(
    model_args: ModelArguments,
    data_args: DataArguments,
    graphs: list,
    tokenizer: PreTrainedTokenizerBase,
    save_dir: Path,
    local_rank: int,
    global_rank: int,
    world_size: int,
    device: torch.device,
    verbose: bool = False,
) -> None:
    assert isinstance(model_args, LeGTJEPAArguments)

    if global_rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f'Pretraining pool: {len(graphs)} subgraphs')
        logging.info(f'World size: {world_size}')
        logging.info(
            f'Effective (global) batch size: {model_args.batch_size * world_size}'
        )

    # Shard subgraphs across ranks
    all_idx = torch.arange(len(graphs))
    idx_this_rank = all_idx.tensor_split(world_size)[global_rank]
    graphs_this_rank = [graphs[i] for i in idx_this_rank.tolist()]

    loader = DataLoader(
        graphs_this_rank,
        batch_size=model_args.batch_size,
        shuffle=True,
        num_workers=5,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True,  # keep BatchNorm / SIGReg batch statistics stable
    )
    if global_rank == 0:
        logging.info(f'Pre-train loader loaded.')

    legtjepa = LeGTJEPA(model_args)
    # This should be false in all cases, but left for ablations.
    if model_args.freeze_text_projection:
        # The h_t branch is dropped from the loss in this ablation, so the
        # text predictor never receives gradient. Freeze it explicitly —
        # otherwise DDP with find_unused_parameters=False crashes on its
        # unused-but-trainable parameters.
        for param in legtjepa.text_predictor.parameters():
            param.requires_grad = False
    # Projections and predictors use BatchNorm; sync statistics across the
    # per-rank batches so SIGReg sees consistent normalization.
    # TODO: Check is this learnable? Is it supposed to be?
    legtjepa = torch.nn.SyncBatchNorm.convert_sync_batchnorm(legtjepa)
    legtjepa.to(device=device)

    model: DDP = DDP(
        legtjepa,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )
    criterion = LeGTJEPALoss(model_args).to(device)
    if global_rank == 0:
        n_trainable = sum(p.numel() for p in model.module.trainable_parameters())
        logging.info(f'Model loaded. Trainable parameters: {n_trainable}')

    optimizer = torch.optim.AdamW(
        model.module.trainable_parameters(),
        lr=model_args.lr,
        weight_decay=model_args.weight_decay,
    )
    total_steps = len(loader) * model_args.epochs
    scheduler = warmup_cosine(optimizer, model_args.warmup_steps, total_steps)

    start_epoch = 1
    ckpt_path = save_dir / 'legtjepa.pt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.module.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        if global_rank == 0:
            logging.info(f'Resuming from {ckpt_path} at epoch {start_epoch}')

    epoch_pbar = tqdm(
        range(start_epoch, model_args.epochs + 1),
        total=model_args.epochs,
        initial=start_epoch - 1,
        desc='Epochs',
        disable=(global_rank != 0),
    )

    for epoch in epoch_pbar:
        epoch_loss = train_epoch(
            model=model,
            criterion=criterion,
            loader=loader,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            model_args=model_args,
            data_args=data_args,
            device=device,
            global_rank=global_rank,
            epoch=epoch,
            log_every_steps=model_args.log_every_steps,
            max_text_length=model_args.max_text_length,
            verbose=verbose,
        )

        if global_rank == 0:
            epoch_pbar.set_postfix({'epoch_loss': f'{epoch_loss:.4f}'})
            logging.info(f'Epoch: {epoch:02d}, Loss: {epoch_loss:.4f}')
            if verbose:
                wandb.log({'train/epoch_loss': epoch_loss, 'train/epoch': epoch})
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch_loss': epoch_loss,
                    'world_size': world_size,
                },
                ckpt_path,
            )
        dist.barrier()
    epoch_pbar.close()


@record
def main() -> None:
    local_rank, global_rank, world_size, device = setup_distributed()

    try:
        root = get_root_dir()
        args = parser.parse_args()
        config_file_path = root / args.config_file
        meta_args, experiment_args = parse_args(config_file_path)

        sweep_id = os.environ.get('WANDB_SWEEP_ID')
        use_wandb = meta_args.verbose or sweep_id is not None

        payload: list = [{}, '']

        if use_wandb and global_rank == 0:
            mode: Literal['online', 'offline'] = (
                'online'
                if sweep_id is not None
                else (
                    'offline' if getattr(meta_args, 'wand_offline', True) else 'online'
                )
            )
            wandb.init(
                project=getattr(meta_args, 'wandb_project', 'legtjepa-pretrain'),
                name=getattr(meta_args, 'wandb_run_name', None),
                config={
                    'world_size': world_size,
                    'global_seed': meta_args.global_seed,
                },
                mode=mode,
            )
            time.sleep(5)  # Helpful for a potential 409 error on wandb servers.
            assert wandb.run is not None
            payload = [dict(wandb.config), wandb.run.id]

        if sweep_id is not None:
            dist.broadcast_object_list(payload, src=0)
        sweep_overrides, run_id = payload

        seed_everything(meta_args.global_seed)

        if global_rank == 0:
            setup_logging(meta_args.log_file_path)
        else:
            logging.basicConfig(level=logging.WARNING)

        root_dir = Path(str(meta_args.root_dir))

        for experiment, experiment_arg in experiment_args.exp_args.items():
            if global_rank == 0:
                logging.info(f'\n***Running*** {experiment}')
            model_args = experiment_arg.model_args
            data_args = experiment_arg.data_args
            assert isinstance(model_args, LeGTJEPAArguments)

            for key, value in sweep_overrides.items():
                if hasattr(model_args, key):
                    setattr(model_args, key, value)
                elif global_rank == 0 and key not in ('world_size', 'global_seed'):
                    logging.warning(
                        f'sweep override {key!r} is not a model field; ignored.'
                    )

            exp_name = f'{experiment}--{run_id}' if sweep_id is not None else experiment

            tokenizer = AutoTokenizer.from_pretrained(model_args.text_model_id)
            graphs = load_source_graphs(meta_args, model_args)

            run_legtjepa(
                model_args=model_args,
                data_args=data_args,
                graphs=graphs,
                tokenizer=tokenizer,
                save_dir=root_dir / 'weights' / exp_name,
                local_rank=local_rank,
                global_rank=global_rank,
                world_size=world_size,
                device=device,
                verbose=use_wandb,
            )
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
