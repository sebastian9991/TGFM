"""Convergence comparison of GramJEPA against GraphCLIP on the GraphCLIP
subgraph-summary pairs, under one training budget.

Launch with:
    torchrun --standalone --nproc_per_node=4 \
        tgfm/experiments/leGTjepa/convergence_main.py \
        --config-file configs/legtjepa/conv/gramjepa_conv_lr1.0e-4.yaml

Arms, selected by ``align_objective`` alone:

    'mse'      LeGTJEPA   cross-modal MSE + SIGReg          (LeGTJEPALoss)
    'volume'   GramJEPA   Gramian volume + SIGReg           (LeGTJEPAVolumeLoss)
    'infonce'  GraphCLIP  models.GraphCLIP from random init, frozen MiniLM text
                          tower, symmetric InfoNCE over the all-gathered global
                          batch                              (GraphCLIPLoss)

Convergence (tgfm/utils/convergence.py). Training losses are not comparable
across arms, so the run evaluates quantities computed identically for every
arm -- target zero-shot macro accuracy, retrieval over a fixed held-out set of
source pairs, RankMe of the graph embeddings -- at step 0, log-spaced steps and
once per epoch, against three clocks: step, samples, and train_s (training
wall clock, paused during evaluation, checkpointing and barriers). Each eval
point writes a ``CONV`` log line; tgfm/tools/collect_convergence.py reads them.

Shared by construction across arms: data, held-out probe set, tokenizer and
max_text_length, per-rank batch_size, graph encoder depth and width, schedule
shape and horizon, augmentation flags, SyncBatchNorm + DDP, sharding, and the
per-epoch step count. The TIMING-CONFIG line records what differs.
"""

import argparse
import logging
import math
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
import torch.distributed as dist
import wandb
from torch import Tensor, nn
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tgfm.evaluation.graphclip_mm_adapter import _Identity, graphclip_on_path
from tgfm.models.legtjepa import LeGTJEPA
from tgfm.models.losses.graphclip_loss import GraphCLIPLoss
from tgfm.models.losses.legtjepaloss import LeGTJEPALoss
from tgfm.models.losses.volumeloss import LeGTJEPAVolumeLoss
from tgfm.utils.args import (
    DataArguments,
    LeGTJEPAArguments,
    MetaArguments,
    ModelArguments,
    parse_args,
)
from tgfm.utils.convergence import ConvMonitor, eval_schedule, holdout_split
from tgfm.utils.logger import setup_logging
from tgfm.utils.path import get_root_dir, get_scratch
from tgfm.utils.process import parse_source_data  # from the GraphCLIP repo
from tgfm.utils.seed import seed_everything
from tgfm.utils.timing import StepTimer, config_fingerprint, log_fingerprint
from tgfm.views.augmentations import batch_graph_aug

parser = argparse.ArgumentParser(
    description='Convergence comparison: GramJEPA vs GraphCLIP.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file', type=str, required=True, help='Path to configuration file.'
)

ARMS: Dict[str, str] = {'mse': 'LeGTJEPA', 'volume': 'GramJEPA', 'infonce': 'GraphCLIP'}
Criterion = Union[LeGTJEPALoss, LeGTJEPAVolumeLoss, GraphCLIPLoss]


class GraphCLIPTAG(nn.Module):
    """GraphCLIP from scratch on the TAG summary pairs (their train.py setting).

    Built from GraphCLIP's own ``models.GraphCLIP`` with the graph-tower sizes
    read from the shared config. No released weights are loaded. The text
    tower is frozen by parameter name (everything outside ``graph_model`` and
    ``logit_scale``), its children are held in eval mode, and its forward runs
    under no_grad, matching LeGTJEPA with freeze_text_backbone=True. GPS
    modules that receive no gradient from InfoNCE are frozen so
    DDP(find_unused_parameters=False) does not wait on them.

    Interface parity with LeGTJEPA:
        forward(batch_g, batch_t) -> {'z_g', 'z_p', 'logit_scale'}
        encode_graph(batch) -> (B, 384)
        encode_text(input_ids, attention_mask, token_type_ids=None) -> (K, 384)
        graph_predictor / text_predictor: identity, so every
        zeroshot_direction reduces to cos(z_g, z_t)
    """

    _UNUSED_GPS_MODULES = ('mlp2', 'attn_pool', 'lora_A_mlp', 'lora_B_mlp')
    _MINILM = 'sentence-transformers/all-MiniLM-L6-v2'

    def __init__(self, args: LeGTJEPAArguments) -> None:
        super().__init__()
        if args.text_model_id != self._MINILM:
            raise ValueError(
                f'GraphCLIP arm uses its MiniLM text tower (text_model="tiny"); '
                f'text_model_id={args.text_model_id!r} would give the two arms '
                f'different tokenizers.'
            )
        if args.graph_pe_dim != 8 or args.attn_type != 'multihead':
            logging.warning(
                'GraphCLIP builds GPS with its own pe_dim=8 / multihead attention; '
                'config has graph_pe_dim=%d attn_type=%r, which only the '
                'LeGTJEPA arm reads.',
                args.graph_pe_dim,
                args.attn_type,
            )

        graphclip_on_path()
        from models import GraphCLIP

        self.backbone = GraphCLIP(
            args.graph_in_dim,
            args.graph_hidden_dim,
            args.graph_num_layers,
            {'dropout': args.attn_dropout},
            text_model='tiny',
        )
        if not isinstance(getattr(self.backbone, 'logit_scale', None), nn.Parameter):
            raise RuntimeError('models.GraphCLIP has no logit_scale parameter.')

        n_text_frozen = 0
        for name, param in self.backbone.named_parameters():
            if name == 'logit_scale' or name.startswith('graph_model.'):
                continue
            param.requires_grad = False
            n_text_frozen += param.numel()
        for mod_name in self._UNUSED_GPS_MODULES:
            sub = getattr(self.backbone.graph_model, mod_name, None)
            if sub is not None:
                for param in sub.parameters():
                    param.requires_grad = False

        self.graph_predictor = _Identity()
        self.text_predictor = _Identity()
        logging.info(
            'GraphCLIP from scratch: %d layers, hidden %d, text tower frozen '
            '(%d params).',
            args.graph_num_layers,
            args.graph_hidden_dim,
            n_text_frozen,
        )

    def train(self, mode: bool = True) -> 'GraphCLIPTAG':
        super().train(mode)
        for name, child in self.backbone.named_children():
            if name != 'graph_model':
                child.eval()
        return self

    def encode_graph(self, batch: Any) -> Tensor:
        graph_embs, _center_embs = self.backbone.encode_graph(batch)
        return graph_embs

    def encode_text(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        return self.backbone.encode_text(input_ids, token_type_ids, attention_mask)

    def forward(
        self,
        batch_g: Any,
        batch_t: Optional[Dict[str, Tensor]] = None,
        image_x: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if batch_t is None:
            raise ValueError('GraphCLIP arm needs tokenized summaries (batch_t).')
        z_g = self.encode_graph(batch_g)
        with torch.no_grad():
            z_t = self.encode_text(
                batch_t['input_ids'],
                batch_t['attention_mask'],
                batch_t.get('token_type_ids'),
            )
        return {'z_g': z_g, 'z_p': z_t, 'logit_scale': self.backbone.logit_scale}

    def trainable_parameters(self) -> Any:
        return (p for p in self.parameters() if p.requires_grad)


def setup_distributed() -> Tuple[int, int, int, torch.device, dist.ProcessGroup]:
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
    cpu_pg = dist.new_group(backend='gloo', timeout=timedelta(hours=1))
    return local_rank, global_rank, world_size, device, cpu_pg


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


def build_scheduler(
    optimizer: torch.optim.Optimizer, model_args: LeGTJEPAArguments, total_steps: int
) -> LambdaLR:
    if model_args.lr_schedule == 'constant':
        return LambdaLR(optimizer, lambda _: 1.0)
    return warmup_cosine(optimizer, model_args.warmup_steps, total_steps)


def build_model(model_args: LeGTJEPAArguments) -> Tuple[nn.Module, str]:
    """(model, checkpoint file name) for the selected arm."""
    if model_args.align_objective == 'infonce':
        return GraphCLIPTAG(model_args), 'graphclip.pt'

    legtjepa = LeGTJEPA(model_args)
    if model_args.freeze_text_projection:
        for param in legtjepa.text_predictor.parameters():
            param.requires_grad = False
    return legtjepa, 'legtjepa.pt'


def build_criterion(model_args: LeGTJEPAArguments) -> Criterion:
    if model_args.align_objective == 'infonce':
        return GraphCLIPLoss()
    if model_args.align_objective == 'volume':
        return LeGTJEPAVolumeLoss(model_args)
    return LeGTJEPALoss(model_args)


def scalar_logs(losses: Dict[str, Tensor]) -> Dict[str, float]:
    """Every scalar the criterion returns; key sets differ across arms."""
    return {
        k: float(v.detach().item())
        for k, v in losses.items()
        if torch.is_tensor(v) and v.numel() == 1
    }


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
    criterion: Criterion,
    loader: DataLoader,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.AdamW,
    scheduler: LambdaLR,
    model_args: LeGTJEPAArguments,
    device: torch.device,
    global_rank: int,
    epoch: int,
    verbose: bool,
    timer: StepTimer,
    steps_per_epoch: int,
    step_offset: int,
    conv: ConvMonitor,
) -> Tuple[float, Dict[str, float]]:
    """One pass over the rank-local shard; returns (mean loss, timing stats)."""
    model.train()
    loss_sum = torch.zeros((), device=device)
    num_batches = 0
    timer.epoch_begin()

    pbar = tqdm(
        loader,
        total=steps_per_epoch,
        desc=f'Epoch {epoch}',
        disable=(global_rank != 0),
        smoothing=0.1,
        leave=False,
    )

    for step, batch in enumerate(pbar):
        # Every rank runs exactly steps_per_epoch steps, so every rank reaches
        # the same global steps and the in-loop eval barriers line up.
        if step >= steps_per_epoch:
            break
        timer.step_begin()
        optimizer.zero_grad(set_to_none=True)

        batch_t = tokenizer(
            batch.summary,
            truncation=True,
            padding=True,
            max_length=model_args.max_text_length,
            return_tensors='pt',
        )
        batch = batch.to(device, non_blocking=True)
        batch_t = {k: v.to(device, non_blocking=True) for k, v in batch_t.items()}

        if model_args.graph_aug:
            batch = batch_graph_aug(
                batch, model_args.aug_feat_drop, model_args.aug_edge_drop
            )
        timer.mark('data')

        if not model_args.adversarial:
            out = model(batch, batch_t)
            timer.mark('forward')
            losses = criterion(out)
            timer.mark('loss')
            losses['loss'].backward()
            timer.mark('backward')
        else:
            timer.mark('forward')
            timer.mark('loss')
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
            timer.mark('backward')

        optimizer.step()
        scheduler.step()
        timer.mark('optim')
        timer.step_end(n_items=int(batch.num_graphs))

        loss_sum += losses['loss'].detach()
        num_batches += 1

        if global_rank == 0 and (step + 1) % model_args.log_every_steps == 0:
            logs = scalar_logs(losses)
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({'loss': f'{logs["loss"]:.4f}', 'lr': f'{lr:.2e}'})
            logging.info(
                f'[epoch {epoch} step {step + 1}/{steps_per_epoch}] '
                + ' '.join(f'{k}={v:.4f}' for k, v in logs.items())
                + f' lr={lr:.2e}'
            )
            if verbose:
                wandb.log(
                    {f'train/{k}': v for k, v in logs.items()}
                    | {'train/epoch': epoch, 'train/lr': lr}
                )

        conv.maybe_eval(step_offset + step + 1, epoch)

    timing = timer.epoch_end()
    loss_tensor = loss_sum / max(1, num_batches)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    return loss_tensor.item(), timing


def run(
    model_args: ModelArguments,
    data_args: DataArguments,
    graphs: list,
    tokenizer: PreTrainedTokenizerBase,
    save_dir: Path,
    local_rank: int,
    global_rank: int,
    world_size: int,
    device: torch.device,
    cpu_pg: dist.ProcessGroup,
    verbose: bool = False,
) -> None:
    assert isinstance(model_args, LeGTJEPAArguments)
    if not model_args.conv_eval:
        raise ValueError('convergence_main.py requires conv_eval: true in the config.')
    objective = model_args.align_objective
    if objective not in ARMS:
        raise ValueError(f'align_objective must be one of {list(ARMS)}, got {objective!r}')
    arm = ARMS[objective]

    # Held out before sharding with a fixed seed: every arm and LR probes the
    # same pairs and none of them trains on those pairs.
    train_idx, hold_idx = holdout_split(len(graphs), model_args.conv_holdout_pairs)
    holdout_graphs = [graphs[i] for i in hold_idx]
    graphs = [graphs[i] for i in train_idx]

    if global_rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f'Arm: {arm} (align_objective={objective})')
        logging.info(
            f'Pretraining pool: {len(graphs)} subgraphs '
            f'(+{len(holdout_graphs)} held out for convergence probes)'
        )
        logging.info(f'World size: {world_size}')
        logging.info(
            f'Effective (global) batch size: {model_args.batch_size * world_size}'
        )

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
        drop_last=True,
    )

    # tensor_split can leave ranks one sample apart, hence len(loader) one step
    # apart; a rank with an extra step would drift out of the eval barriers.
    n_steps = torch.tensor(len(loader), device=device)
    dist.all_reduce(n_steps, op=dist.ReduceOp.MIN)
    steps_per_epoch = int(n_steps.item())

    net, ckpt_name = build_model(model_args)
    # Both arms: SyncBatchNorm adds a collective per BatchNorm layer.
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.to(device=device)

    model: DDP = DDP(
        net,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )
    criterion = build_criterion(model_args).to(device)

    n_trainable = sum(p.numel() for p in model.module.trainable_parameters())
    if global_rank == 0:
        logging.info(f'Model loaded. Trainable parameters: {n_trainable}')
        logging.info(f'LR schedule: {model_args.lr_schedule}')
        fp = config_fingerprint(model_args, world_size, n_trainable)
        fp['arm'] = arm
        fp['sync_bn'] = True
        fp['lr_schedule'] = model_args.lr_schedule
        fp['lr'] = model_args.lr
        fp['warmup_steps'] = model_args.warmup_steps
        fp['epochs'] = model_args.epochs
        fp['steps_per_epoch'] = steps_per_epoch
        log_fingerprint(fp)

    timer = StepTimer(
        device,
        world_size=world_size,
        sample_every=getattr(model_args, 'timing_sample_every', 50),
        warmup=getattr(model_args, 'timing_warmup_steps', 10),
        enabled=getattr(model_args, 'timing_enabled', True),
    )

    optimizer = torch.optim.AdamW(
        model.module.trainable_parameters(),
        lr=model_args.lr,
        weight_decay=model_args.weight_decay,
    )
    total_steps = steps_per_epoch * model_args.epochs
    scheduler = build_scheduler(optimizer, model_args, total_steps)

    start_epoch = 1
    ckpt_path = save_dir / ckpt_name
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.module.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        if global_rank == 0:
            logging.info(f'Resuming from {ckpt_path} at epoch {start_epoch}')
            logging.warning(
                'Resumed convergence run: clock/train_s restarts at 0, so the '
                'curve is not comparable. Clear the weights dir and rerun.'
            )

    conv = ConvMonitor(
        model=model.module,
        tokenizer=tokenizer,
        model_args=model_args,
        data_args=data_args,
        holdout_graphs=holdout_graphs,
        schedule=eval_schedule(
            total_steps,
            model_args.conv_eval_first_step,
            model_args.conv_eval_every_steps,
        ),
        device=device,
        global_rank=global_rank,
        world_size=world_size,
        cpu_pg=cpu_pg,
        verbose=verbose,
    )
    if start_epoch == 1:
        conv.clock.start()
        conv.maybe_eval(0, epoch=0)  # random-init anchor of every curve
        conv.clock.pause()

    epoch_pbar = tqdm(
        range(start_epoch, model_args.epochs + 1),
        total=model_args.epochs,
        initial=start_epoch - 1,
        desc='Epochs',
        disable=(global_rank != 0),
    )

    total_train_s, epochs_timed = 0.0, 0
    for epoch in epoch_pbar:
        conv.eval_s = 0.0
        conv.clock.start()
        epoch_loss, timing = train_epoch(
            model=model,
            criterion=criterion,
            loader=loader,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            model_args=model_args,
            device=device,
            global_rank=global_rank,
            epoch=epoch,
            verbose=verbose,
            timer=timer,
            steps_per_epoch=steps_per_epoch,
            step_offset=(epoch - 1) * steps_per_epoch,
            conv=conv,
        )
        conv.clock.pause()
        # StepTimer's epoch clock ran through the in-epoch evaluations.
        timing['eval_in_epoch_s'] = conv.eval_s
        timing['epoch_train_s'] = timing['epoch_s'] - conv.eval_s
        total_train_s += timing['epoch_train_s']
        epochs_timed += 1

        if global_rank == 0:
            epoch_pbar.set_postfix({'epoch_loss': f'{epoch_loss:.4f}'})
            logging.info(f'Epoch: {epoch:02d}, Loss: {epoch_loss:.4f}')
            logging.info(
                f'TIMING epoch {epoch:02d}: {StepTimer.format(timing)} '
                f'eval_in_epoch={conv.eval_s:.1f}s '
                f'epoch_train={timing["epoch_train_s"]:.1f}s'
            )
            if verbose:
                wandb.log({'train/epoch_loss': epoch_loss, 'train/epoch': epoch})
                wandb.log(
                    {f'time/{k}': v for k, v in timing.items()}
                    | {'train/epoch': epoch}
                )
            torch.save(
                {
                    'epoch': epoch,
                    'arm': arm,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch_loss': epoch_loss,
                    'world_size': world_size,
                },
                ckpt_path,
            )
        dist.barrier(group=cpu_pg)
    epoch_pbar.close()

    if global_rank == 0 and epochs_timed > 0:
        logging.info(
            f'TIMING-TOTAL arm={arm} train_s={total_train_s:.1f} '
            f'epochs={epochs_timed} mean_epoch_s={total_train_s / epochs_timed:.1f} '
            f'conv_clock_s={conv.clock.seconds:.1f}'
        )
        if verbose and wandb.run is not None:
            wandb.run.summary['time/total_train_s'] = total_train_s
            wandb.run.summary['time/mean_epoch_s'] = total_train_s / epochs_timed


@record
def main() -> None:
    local_rank, global_rank, world_size, device, cpu_pg = setup_distributed()

    try:
        root = get_root_dir()
        args = parser.parse_args()
        config_file_path = root / args.config_file
        meta_args, experiment_args = parse_args(config_file_path)

        sweep_id = os.environ.get('WANDB_SWEEP_ID')
        use_wandb = meta_args.verbose or sweep_id is not None
        if sweep_id is not None:
            assert meta_args.verbose, 'sweeps require verbose: true'

        payload: list = [{}, '']

        if use_wandb and global_rank == 0:
            mode: Literal['online', 'offline'] = (
                'online'
                if sweep_id is not None
                else (
                    'offline' if getattr(meta_args, 'wandb_offline', True) else 'online'
                )
            )
            logging.info(f'Using wandb {mode}.')
            wandb.init(
                project=getattr(meta_args, 'wandb_project', 'legtjepa-convergence'),
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

            run(
                model_args=model_args,
                data_args=data_args,
                graphs=graphs,
                tokenizer=tokenizer,
                save_dir=root_dir / 'weights' / exp_name,
                local_rank=local_rank,
                global_rank=global_rank,
                world_size=world_size,
                device=device,
                cpu_pg=cpu_pg,
                verbose=use_wandb,
            )
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
