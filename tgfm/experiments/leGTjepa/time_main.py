"""Distributed pretraining on the GraphCLIP subgraph-summary pairs, instrumented
for end-to-end timing of GramJEPA against GraphCLIP.

Launch with:
    torchrun --standalone --nproc_per_node=4 \
        tgfm/experiments/leGTjepa/main.py \
        --config-file configs/legtjepa/timing/gramjepa_timing.yaml

Arms, selected by ``align_objective`` alone, so that two timing runs differ in
exactly one config field:

    'mse'      LeGTJEPA   cross-modal MSE + SIGReg          (LeGTJEPALoss)
    'volume'   GramJEPA   Gramian volume + SIGReg           (LeGTJEPAVolumeLoss)
    'infonce'  GraphCLIP  models.GraphCLIP from random init, frozen MiniLM text
                          tower, symmetric InfoNCE over the all-gathered global
                          batch                              (GraphCLIPLoss)

Shared by construction across arms: data (parse_source_data), tokenizer and
max_text_length, per-rank batch_size, graph_num_layers, graph_hidden_dim,
graph_in_dim, attn_dropout, graph_aug / adversarial flags, SyncBatchNorm + DDP
wrapping, sharding, loader, and the timer. What differs is the objective and
the heads each method needs: GramJEPA carries projections and predictors,
GraphCLIP carries its GPS ``mlp`` head. ``trainable_params`` in TIMING-CONFIG
records that difference rather than hiding it.

Timing attribution. ``loss_ms`` is the criterion forward, which is the whole
objective and not only the alignment term: for GramJEPA it includes SIGReg and
the all_gather feeding it, for GraphCLIP the all_gather of z_g and z_p feeding
the global logits. Backward of the objective is fused into ``backward_ms``.
The isolated benchmark (tools/bench_alignment_cost.py) remains the instrument
for the per-term complexity claim.

Expects GraphCLIP's released data layout under ``meta_args.root_dir`` and the
GraphCLIP repo at GRAPHCLIP_ROOT (default third_party/GraphCLIP) for the
'infonce' arm.
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
from tgfm.evaluation.zero_shot_eval import zeroshot_macro
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
from tgfm.utils.logger import setup_logging
from tgfm.utils.path import get_root_dir, get_scratch
from tgfm.utils.process import parse_source_data  # from the GraphCLIP repo
from tgfm.utils.seed import seed_everything
from tgfm.utils.timing import StepTimer, config_fingerprint, log_fingerprint
from tgfm.views.augmentations import batch_graph_aug

parser = argparse.ArgumentParser(
    description='Distributed pretraining: LeGTJEPA / GramJEPA / GraphCLIP.',
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
    read from the shared config, so the encoder depth and width match the
    GramJEPA run being compared. No released weights are loaded.

    Text tower: frozen, as in GraphCLIP (freeze_text). Frozen here by name
    rather than through a method on their class: every parameter outside
    ``graph_model`` and ``logit_scale`` is locked, and the non-graph children
    are held in eval mode. The text forward runs under no_grad, matching
    LeGTJEPA.encode_text with freeze_text_backbone=True, so neither arm pays
    for an autograd graph through MiniLM.

    Unused GPS modules (mlp2, attn_pool, LoRA pair) run in forward or serve
    prompt tuning but receive no gradient from InfoNCE; they are frozen so
    DDP(find_unused_parameters=False) does not wait on them. Same set as
    GraphCLIPMM.

    Interface parity with LeGTJEPA for train_epoch and zeroshot_macro:
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
                'LeGTJEPA arm reads. Match them for a timing pair.',
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
        return LambdaLR(optimizer, lambda _: 1.0)  # GraphCLIP train.py: no scheduler
    return warmup_cosine(optimizer, model_args.warmup_steps, total_steps)


def build_model(model_args: LeGTJEPAArguments) -> Tuple[nn.Module, str]:
    """(model, checkpoint file name) for the selected arm."""
    if model_args.align_objective == 'infonce':
        return GraphCLIPTAG(model_args), 'graphclip.pt'

    legtjepa = LeGTJEPA(model_args)
    if model_args.freeze_text_projection:
        # The h_t branch is dropped from the loss in this ablation, so the
        # text predictor never receives gradient; freeze it for DDP.
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
) -> Tuple[float, Dict[str, float]]:
    """One pass over the rank-local shard; returns (mean loss, timing stats)."""
    model.train()
    # Accumulated on device: a per-step .item() is a host sync on every step,
    # which serialises CPU and GPU in both arms and inflates epoch_s.
    loss_sum = torch.zeros((), device=device)
    num_batches = 0
    timer.epoch_begin()

    pbar = tqdm(
        loader,
        total=len(loader),
        desc=f'Epoch {epoch}',
        disable=(global_rank != 0),
        smoothing=0.1,
        leave=False,
    )

    for step, batch in enumerate(pbar):
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
            # m fused forward/backward passes: the breakdown collapses into
            # backward_ms; step_ms and epoch_s stay exact.
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
                f'[epoch {epoch} step {step + 1}/{len(loader)}] '
                + ' '.join(f'{k}={v:.4f}' for k, v in logs.items())
                + f' lr={lr:.2e}'
            )
            if verbose:
                wandb.log(
                    {f'train/{k}': v for k, v in logs.items()}
                    | {'train/epoch': epoch, 'train/lr': lr}
                )

    # epoch_end reduces across ranks itself, so every rank must reach it.
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
    objective = model_args.align_objective
    if objective not in ARMS:
        raise ValueError(
            f'align_objective must be one of {list(ARMS)}, got {objective!r}'
        )
    arm = ARMS[objective]

    if global_rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f'Arm: {arm} (align_objective={objective})')
        logging.info(f'Pretraining pool: {len(graphs)} subgraphs')
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
        drop_last=True,  # keep BatchNorm / SIGReg batch statistics stable
    )

    net, ckpt_name = build_model(model_args)
    # Applied to both arms: SyncBatchNorm adds a collective per BatchNorm layer,
    # so converting only one arm would put a communication confound into the
    # timing comparison.
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
        log_fingerprint(fp)

    timer = StepTimer(
        device,
        world_size=world_size,
        sample_every=getattr(model_args, 'timing_sample_every', 50),
        warmup=getattr(model_args, 'timing_warmup_steps', 10),
        enabled=getattr(model_args, 'timing_enabled', True),
    )
    timer.plan(
        len(loader), target_samples=int(os.environ.get('TGFM_TIMING_SAMPLES', 24))
    )

    optimizer = torch.optim.AdamW(
        model.module.trainable_parameters(),
        lr=model_args.lr,
        weight_decay=model_args.weight_decay,
    )
    total_steps = len(loader) * model_args.epochs
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

    epoch_pbar = tqdm(
        range(start_epoch, model_args.epochs + 1),
        total=model_args.epochs,
        initial=start_epoch - 1,
        desc='Epochs',
        disable=(global_rank != 0),
    )

    # Sum of epoch_s over epochs run in this process: excludes eval,
    # checkpointing and the barrier, which sit outside train_epoch.
    total_train_s, epochs_timed = 0.0, 0
    for epoch in epoch_pbar:
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
        )
        total_train_s += timing['epoch_s']
        epochs_timed += 1

        if global_rank == 0:
            epoch_pbar.set_postfix({'epoch_loss': f'{epoch_loss:.4f}'})
            logging.info(f'Epoch: {epoch:02d}, Loss: {epoch_loss:.4f}')
            logging.info(f'TIMING epoch {epoch:02d}: {StepTimer.format(timing)}')
            if verbose:
                wandb.log({'train/epoch_loss': epoch_loss, 'train/epoch': epoch})
                wandb.log(
                    {f'time/{k}': v for k, v in timing.items()} | {'train/epoch': epoch}
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
        if epoch % data_args.eval_every_epochs == 0 or epoch == model_args.epochs:
            if verbose and global_rank == 0:
                logging.info('Evaluation...')
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
                logging.info(f'{ {k: f"{100 * v:.2f}" for k, v in per_ds.items()} }')
                wandb.log(
                    {
                        'eval/zeroshot_macro': macro,
                        **{f'eval/zeroshot_{k}': float(v) for k, v in per_ds.items()},
                        'train/epoch': epoch,
                    }
                )
        dist.barrier(group=cpu_pg)
    epoch_pbar.close()

    if global_rank == 0 and epochs_timed > 0:
        mean_epoch_s = total_train_s / epochs_timed
        logging.info(
            f'TIMING-TOTAL arm={arm} train_s={total_train_s:.1f} '
            f'epochs={epochs_timed} mean_epoch_s={mean_epoch_s:.1f} '
            f'(first epoch {start_epoch}; resumed runs count only epochs run here)'
        )
        if verbose and wandb.run is not None:
            wandb.run.summary['time/total_train_s'] = total_train_s
            wandb.run.summary['time/mean_epoch_s'] = mean_epoch_s
            wandb.run.summary['time/epochs_timed'] = epochs_timed


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
