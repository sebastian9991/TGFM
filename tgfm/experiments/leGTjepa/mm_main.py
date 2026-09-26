"""Distributed MM-Graph pretraining for three-tower LeGTJEPA.

Pretrains on the MM-Graph link-prediction graphs (no node labels needed --
the objective uses none) and evaluates transfer to the held-out node
classification graphs by a linear probe on frozen embeddings, every
``data_args.eval_every_epochs`` epochs.

    pretrain : sports-copurchase + cloth-copurchase + books-lp
    probe    : ele-fashion + books-nc

Table rows (model_args.model selects the method):
    GramJEPA  model='LeGTJEPA', align_objective='volume'
        G-T     use_text=True   use_image=False  graph_feat='text'
        G-I     use_text=False  use_image=True   graph_feat='image'
        G-T-I   use_text=True   use_image=True   graph_feat='text' | 'text_image'
    GraphCLIP model='GraphCLIP' (from scratch, GraphCLIP GPS + InfoNCE)
        G-T     use_text=True   use_image=False  graph_feat='text'
        G-I     use_text=False  use_image=True   graph_feat='image'

Launch with:
    torchrun --standalone --nproc_per_node=4 \
        tgfm/experiments/legtjepa/mm_main.py \
        --config-file configs/volume_mmgraph.yaml.
"""

import argparse
import logging
import math
import os
import time
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Literal, Tuple, Union

import torch
import torch.distributed as dist
import wandb
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, Subset
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from tgfm.dataset.evaluation.mm_load import load_mm_data
from tgfm.evaluation.graphclip_mm_adapter import GraphCLIPMM
from tgfm.evaluation.mm_linear_probe import evaluate_dataset as probe_dataset
from tgfm.evaluation.mm_lp_linear_probe import evaluate_dataset as lp_probe_dataset
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
from tgfm.utils.mm_features import (
    center_text_feature,
    check_graph_features,
    select_graph_features,
)
from tgfm.utils.mm_sampler import parse_mm_target_data
from tgfm.utils.path import get_root_dir
from tgfm.utils.seed import seed_everything
from tgfm.views.augmentations import batch_graph_aug

parser = argparse.ArgumentParser(
    description='Distributed pretraining MM LeGTJEPA.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file', type=str, required=True, help='Path to configuration file.'
)


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
    # Generous timeout: rank 0 embeds the whole target graph during the probe
    # while the other ranks wait here. books-nc is 685K nodes.
    cpu_pg = dist.new_group(backend='gloo', timeout=timedelta(hours=4))
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


def load_source_graphs(
    meta_args: MetaArguments, model_args: ModelArguments
) -> ConcatDataset:
    """Ego-subgraphs from the MM-Graph pretraining datasets.

    One subgraph per node. Node features are chosen by model_args.graph_feat
    before the sampler is built; the center node's image feature rides along
    as image_x -- one positive tuple per node.
    """
    assert isinstance(model_args, LeGTJEPAArguments)
    parts = []
    for name in model_args.source_data.split('+'):
        data, _, _ = load_mm_data(name, feat_name=model_args.mm_feat_name)
        data = select_graph_features(data, model_args)
        ds = parse_mm_target_data(name, data)
        ds.assert_alignment()  # item u is node u: keeps the modalities aligned
        check_graph_features(ds, data)
        parts.append(ds)
        logging.info(f'Loaded source dataset {name} ({len(ds)} subgraphs)')
    graphs = ConcatDataset(parts)
    logging.info(f'Pretraining pool: {len(graphs)} subgraphs')
    return graphs


def train_epoch(
    model: DDP,
    criterion: Union[LeGTJEPALoss, LeGTJEPAVolumeLoss, GraphCLIPLoss],
    loader: DataLoader,
    optimizer: torch.optim.AdamW,
    scheduler: LambdaLR,
    model_args: ModelArguments,
    data_args: DataArguments,
    device: torch.device,
    cpu_pg: dist.ProcessGroup,
    global_rank: int,
    epoch: int,
    log_every_steps: int,
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

        batch = batch.to(device, non_blocking=True)
        # Precomputed features: batch.x is the graph node feature (per
        # graph_feat), batch.image_x the center image feature. No tokenizer.
        # Partner targets are read *before* augmentation: augmentation is a
        # graph-side view, and perturbing the targets as well would make the
        # alignment term chase a moving target on both sides. (Previously the
        # text target was indexed after batch_graph_aug, so feature drop leaked
        # into it whenever graph_aug=True.)
        text_x = center_text_feature(batch, model_args) if model_args.use_text else None
        image_x = batch.image_x if model_args.use_image else None

        if model_args.graph_aug:
            batch = batch_graph_aug(
                batch, model_args.aug_feat_drop, model_args.aug_edge_drop
            )

        if not model_args.adversarial:
            losses = criterion(model(batch, text_x, image_x=image_x))
            losses['loss'].backward()
        else:
            m, eps = model_args.adv_steps, model_args.adv_step_size
            x_clean = batch.x
            perturb = torch.empty_like(x_clean).uniform_(-eps, eps).requires_grad_()
            for i in range(m):
                batch.x = x_clean + perturb
                losses = criterion(model(batch, text_x, image_x=image_x))
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

        if global_rank == 0 and (step + 1) % log_every_steps == 0:
            # The two arms return different key sets (the volume arm adds
            # volume / cos_gt / cos_gi / cos_ti / sigreg_image), so log
            # whatever the criterion produced rather than a fixed list.
            scalars = {k: v.item() for k, v in losses.items()}
            postfix = {'loss': f'{scalars["loss"]:.4f}'}
            for key, short in (
                ('cross', 'cross'),
                ('infonce', 'nce'),
                ('logit_scale', 'scale'),
                ('sigreg_graph', 'sg_g'),
                ('sigreg_text', 'sg_t'),
                ('sigreg_image', 'sg_i'),
            ):
                if key in scalars:  # methods and modality rows return different keys
                    postfix[short] = f'{scalars[key]:.4f}'
            postfix['lr'] = f'{scheduler.get_last_lr()[0]:.2e}'
            pbar.set_postfix(postfix)
            logging.info(
                f'[epoch {epoch} step {step + 1}/{len(loader)}] '
                + ' '.join(f'{k}={v:.4f}' for k, v in scalars.items())
                + f' lr={scheduler.get_last_lr()[0]:.2e}'
            )
            if verbose:
                wandb.log(
                    {f'train/{k}': v for k, v in scalars.items()}
                    | {
                        'train/epoch': epoch,
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
    graphs: ConcatDataset,
    save_dir: Path,
    local_rank: int,
    global_rank: int,
    world_size: int,
    device: torch.device,
    cpu_pg: dist.ProcessGroup,
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
    graphs_this_rank = Subset(graphs, idx_this_rank.tolist())

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
        logging.info('Pre-train loader loaded.')

    is_graphclip = model_args.model == 'GraphCLIP'
    legtjepa: torch.nn.Module = (
        GraphCLIPMM(model_args) if is_graphclip else LeGTJEPA(model_args)
    )
    # A frozen partner projection drops that branch's anchor from the loss, so
    # its predictor never receives gradient. Freeze it explicitly -- otherwise
    # DDP with find_unused_parameters=False crashes on its unused-but-trainable
    # parameters.
    if not is_graphclip:
        frozen_branches = []
        if model_args.use_text and model_args.freeze_text_projection:
            frozen_branches.append(('text_predictor', legtjepa.text_predictor))
        if model_args.use_image and model_args.freeze_image_projection:
            frozen_branches.append(('image_predictor', legtjepa.image_predictor))
        for branch_name, module in frozen_branches:
            for param in module.parameters():
                param.requires_grad = False
            logging.info('%s frozen (its anchor is dropped from the loss).', branch_name)
    # Projections and predictors use BatchNorm; sync statistics across the
    # per-rank batches so SIGReg sees consistent normalization.
    legtjepa = torch.nn.SyncBatchNorm.convert_sync_batchnorm(legtjepa)
    legtjepa.to(device=device)

    model: DDP = DDP(
        legtjepa,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )
    criterion: Union[LeGTJEPALoss, LeGTJEPAVolumeLoss, GraphCLIPLoss]
    if is_graphclip:
        criterion = GraphCLIPLoss()
    elif model_args.align_objective == 'volume':
        criterion = LeGTJEPAVolumeLoss(model_args)
    else:
        criterion = LeGTJEPALoss(model_args)
    criterion = criterion.to(device)
    if global_rank == 0:
        n_trainable = sum(p.numel() for p in model.module.trainable_parameters())
        logging.info(f'Model loaded. Trainable parameters: {n_trainable}')
        logging.info(f'Aligning with: {model_args.align_objective}')
        logging.info(f'Criterion: {type(criterion).__name__}')
        logging.info(f'LR schedule: {model_args.lr_schedule}')
        logging.info(f'Probe representation: {model_args.probe_representation}')

    optimizer = torch.optim.AdamW(
        model.module.trainable_parameters(),
        lr=model_args.lr,
        weight_decay=model_args.weight_decay,
    )
    total_steps = len(loader) * model_args.epochs
    scheduler = (
        LambdaLR(optimizer, lambda _: 1.0)  # GraphCLIP train.py: no scheduler
        if model_args.lr_schedule == 'constant'
        else warmup_cosine(optimizer, model_args.warmup_steps, total_steps)
    )

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

    # Probe bookkeeping. NC: select on macro val accuracy over targets and
    # report that epoch's test metrics. LP: this branch still reads only
    # res['test/cosine'], so it selects on test MRR -- move it to a val MRR
    # before reporting LP numbers as selected results.
    best_score, best_epoch = -1.0, 0
    best_res_dict: dict = {}
    best_path = save_dir / 'legtjepa_best.pt'
    # Only restore when resuming: a fresh run in a reused directory must not
    # inherit a stale best from an earlier run (it is overwritten at the
    # first eval, since best_score starts at -1).
    if global_rank == 0 and start_epoch > 1 and best_path.exists():
        best_meta = torch.load(best_path, map_location='cpu', weights_only=False)
        best_score = best_meta['selection_score']
        best_epoch = best_meta['epoch']
        best_res_dict = best_meta['results']
        logging.info(
            f'Restored selection state: best score {best_score:.5f} '
            f'at epoch {best_epoch}'
        )
        del best_meta

    epoch_pbar = tqdm(
        range(start_epoch, model_args.epochs + 1),
        total=model_args.epochs,
        initial=start_epoch - 1,
        desc='Epochs',
        disable=(global_rank != 0),
    )

    if global_rank == 0:
        logging.info(
            f'eval_every_epochs={data_args.eval_every_epochs} '
            f'verbose={verbose} '
            f'wandb_mode={wandb.run.settings.mode if wandb.run else None} '
            f'targets={data_args.target_data.split("+")}'
        )

    for epoch in epoch_pbar:
        epoch_loss = train_epoch(
            model=model,
            criterion=criterion,
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            model_args=model_args,
            data_args=data_args,
            device=device,
            cpu_pg=cpu_pg,
            global_rank=global_rank,
            epoch=epoch,
            log_every_steps=model_args.log_every_steps,
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

        # Linear probe on frozen embeddings (MM-Graph NC targets).
        # probe_dataset restores the model's training mode on exit.
        # Frozen-embedding linear probe. task_name selects the readout:
        #   'node' -> NC accuracy (ele-fashion, books-nc)
        #   'link' -> LP MRR/Hits against the shipped negatives (Table 5)
        # Both probes restore the model's training mode on exit.
        if global_rank == 0 and epoch % data_args.eval_every_epochs == 0:
            logging.info(f'------------Evaluate-{epoch}------------')
            is_lp = data_args.task_name == 'link'
            probe_res = {}
            score_sum, n = 0.0, 0
            for data_name in data_args.target_data.split('+'):
                if is_lp:
                    res = lp_probe_dataset(
                        model.module,
                        data_name,
                        model_args,
                        device,
                        data_args.eval_batch_size,
                    )
                    score = res['test/cosine']['mrr']  # test-selected; see note above
                    probe_res[data_name] = res
                    logging.info(
                        'DATA: {} | METHOD: lp-probe | cosine MRR {:.4f} '
                        'H@1 {:.4f} H@10 {:.4f}'.format(
                            data_name,
                            res['test/cosine']['mrr'],
                            res['test/cosine']['hits@1'],
                            res['test/cosine']['hits@10'],
                        )
                    )
                else:
                    res = probe_dataset(
                        model.module,
                        data_name,
                        model_args,
                        data_args.eval_seeds,
                        data_args.eval_batch_size,
                        device,
                        model_args.mm_feat_name,
                        model_args.probe_representation,
                    )
                    score = res['val/acc'][0]
                    probe_res[data_name] = res
                    logging.info(
                        'DATA: {} | METHOD: nc-probe | VAL-ACC: {:.5f} | '
                        'TEST-ACC: {:.5f}+/-{:.5f} | TEST-F1: {:.5f}+/-{:.5f}'.format(
                            data_name,
                            res['val/acc'][0],
                            *res['test/acc'],
                            *res['test/f1'],
                        )
                    )
                score_sum += score
                n += 1

            ave_score = score_sum / max(1, n)
            metric_name = 'TEST-MRR' if is_lp else 'VAL-ACC'
            logging.info('--------------------------------')
            logging.info(f'AVE-ALL-{metric_name}: {ave_score:.5f}')
            if verbose:
                if is_lp:
                    wandb.log(
                        {
                            f'eval/mrr_{k}': v['test/cosine']['mrr']
                            for k, v in probe_res.items()
                        }
                        | {'eval/probe_macro': ave_score, 'train/epoch': epoch}
                    )
                else:
                    log = {
                        f'eval/{key.replace("/", "_")}_{ds}': m
                        for ds, r in probe_res.items()
                        for key, (m, _) in r.items()
                    }
                    # eval/probe_macro is now macro VAL accuracy, so sweeps
                    # that maximize it tune on validation, not test.
                    log['eval/probe_macro'] = ave_score
                    log['eval/test_macro_acc'] = sum(
                        r['test/acc'][0] for r in probe_res.values()
                    ) / max(1, n)
                    log['eval/test_macro_f1'] = sum(
                        r['test/f1'][0] for r in probe_res.values()
                    ) / max(1, n)
                    log['train/epoch'] = epoch
                    wandb.log(log)
            if ave_score > best_score:
                best_score = ave_score
                best_res_dict = deepcopy(probe_res)
                best_epoch = epoch
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'selection_metric': metric_name,
                        'selection_score': best_score,
                        'results': best_res_dict,
                    },
                    best_path,
                )
            logging.info(
                f'BEST-{metric_name}: {best_score:.5f} | BEST-EPOCH: {best_epoch}'
            )
            logging.info('--------------------------------')
        dist.barrier(group=cpu_pg)

    epoch_pbar.close()

    if global_rank == 0 and best_res_dict:
        logging.info('------------Final-Evaluate-------------')
        logging.info(f'-----Selected Epoch {best_epoch} (by validation)-----')
        if data_args.task_name == 'link':
            for data_name, res in best_res_dict.items():
                logging.info(
                    'DATA: {} | METHOD: lp-probe | cosine MRR {:.4f} H@1 {:.4f} '
                    'H@3 {:.4f} H@10 {:.4f}'.format(
                        data_name,
                        res['test/cosine']['mrr'],
                        res['test/cosine']['hits@1'],
                        res['test/cosine']['hits@3'],
                        res['test/cosine']['hits@10'],
                    )
                )
            logging.info(f'BEST-MACRO-TEST-MRR: {best_score:.5f}')
        else:
            for data_name, res in best_res_dict.items():
                logging.info(
                    'DATA: {} | METHOD: nc-probe | VAL-ACC: {:.5f} | '
                    'TEST-ACC: {:.5f}+/-{:.5f} | TEST-F1: {:.5f}+/-{:.5f}'.format(
                        data_name,
                        res['val/acc'][0],
                        *res['test/acc'],
                        *res['test/f1'],
                    )
                )
            n = len(best_res_dict)
            logging.info(f'SELECTED-MACRO-VAL-ACC: {best_score:.5f}')
            logging.info(
                'SELECTED-MACRO-TEST-ACC: {:.5f} | SELECTED-MACRO-TEST-F1: {:.5f}'.format(
                    sum(r['test/acc'][0] for r in best_res_dict.values()) / n,
                    sum(r['test/f1'][0] for r in best_res_dict.values()) / n,
                )
            )
            # Acc & F1 per target, in target_data order, x100: paste into the table.
            logging.info(
                'TABLE-ROW: '
                + ' & '.join(
                    f"{100 * r['test/acc'][0]:.1f} & {100 * r['test/f1'][0]:.1f}"
                    for r in best_res_dict.values()
                )
            )


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
                elif hasattr(data_args, key):
                    setattr(data_args, key, value)
                elif global_rank == 0 and key not in ('world_size', 'global_seed'):
                    logging.warning(
                        f'sweep override {key!r} is not a model or data field; ignored.'
                    )

            exp_name = f'{experiment}--{run_id}' if sweep_id is not None else experiment

            graphs = load_source_graphs(meta_args, model_args)

            run_legtjepa(
                model_args=model_args,
                data_args=data_args,
                graphs=graphs,
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
