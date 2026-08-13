"""LeGTJEPA pretraining on GSTBench, with per-step diagnostics and wandb sweeps.

Split out from the shared transfer trainer so the loss components can be
logged at step granularity: an epoch is ~11k steps / 2 hours, and a single
end-of-epoch scalar hides whether the cross term or SIGReg is doing the work.

Logged every ``log_step`` steps (running means since the last log):
    total / cross / sigreg_g / sigreg_t
    ratio     (lg*sigreg_g + lt*sigreg_t) / cross -- the weighted balance
    gnorm     pre-clip global grad norm
    erank_g/t effective rank of each embedding matrix

Downstream metrics, logged at every evaluation:
    eval/{dataset}_val, eval/{dataset}_test        per dataset
    eval/macro_val, eval/macro_test                all eval datasets
    eval/in_domain_val, eval/in_domain_test        pretraining domain
    eval/cross_domain_val, eval/cross_domain_test  everything else
The in-domain set comes from ``data_args.in_domain_data_names``; cross-domain
is its complement within ``eval_data_names``. For a transferability claim
``eval/cross_domain_val`` is the sweep metric to maximize -- the in-domain
macro is confounded by papers100M being academic citation data, so it rewards
domain overlap rather than transfer.

Sweeps: rank 0 calls wandb.init, then broadcasts wandb.config to the other
ranks before any of them build a model, so every rank trains the same
configuration. Launch via ``wandb agent``; WANDB_SWEEP_ID is inherited
through mp.spawn.

Differences from the shared trainer, both deliberate:
  - clip_grad_norm_ is called AFTER backward(). In the shared trainer it runs
    before, so with zero_grad(set_to_none=True) every gradient is None at that
    point and the clip silently never happens.
  - eval may run at fractional-epoch cadence via ``eval_every_steps``.
"""

import argparse
import datetime
import gc
import logging
import os
import os.path
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import dgl
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from tgfm.dataset.lazy_graph import LazyGraphDataset
from tgfm.evaluation.transfer_eval import (
    create_k_shot_tasks,
    eval_downstream,
    get_node_data_all,
)
from tgfm.models.base_models.base_models import GATNet, GCNNet
from tgfm.models.pretrain_model.legtjepa_pretrain import PretrainLeGTJEPA
from tgfm.utils.args import (
    DataArguments,
    MetaArguments,
    TransferArguments,
    parse_args,
)
from tgfm.utils.collate import Universal_Collator
from tgfm.utils.logger import setup_logging
from tgfm.utils.lr import ConstantLRScheduler, WarmupCosineDecayScheduler
from tgfm.utils.path import get_root_dir, get_scratch
from tgfm.utils.seed import seed_everything
from tgfm.utils.transfer_utils import init_process_group, worker_init_fn

GraphEncoder = Union[GCNNet, GATNet]

os.environ['DGLBACKEND'] = 'pytorch'

# Fallback split if data_args.in_domain_data_names is unset: papers100M is an
# academic citation graph, so citation datasets are in-domain.
DEFAULT_IN_DOMAIN = ('cora', 'citeseer', 'pubmed', 'dblp', 'arxiv')

parser = argparse.ArgumentParser(
    description='LeGTJEPA pretraining (GSTBench transfer).',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file', type=str, required=True, help='Path to configuration file.'
)
parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs on node.')


def macro(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float('nan')


def report_downstream(
    model: torch.nn.Module,
    nc_data: dict,
    nc_tasks: dict,
    device: torch.device,
    model_args: TransferArguments,
    in_domain: Sequence[str],
    header: str,
    step: int,
    use_wandb: bool,
) -> Tuple[float, float, dict]:
    """Evaluate, log per dataset and per domain split, return (val, test, raw)."""
    logging.info(f'------------{header}------------')
    nc_dict = eval_downstream(model, nc_data, nc_tasks, device, model_args)

    metrics: Dict[str, float] = {}
    val_all: List[float] = []
    test_all: List[float] = []
    val_in: List[float] = []
    test_in: List[float] = []
    val_out: List[float] = []
    test_out: List[float] = []

    for data_name in nc_dict:
        for method_name in nc_dict[data_name]:
            v = nc_dict[data_name][method_name]
            logging.info(
                'DATA: {} | METHOD: {} \nVAL-ACC: {:.5f}±{:.5f} | TEST-ACC: {:.5f}±{:.5f}'.format(
                    data_name, method_name, v[0], v[1], v[2], v[3]
                )
            )
            metrics[f'eval/{data_name}_val'] = float(v[0])
            metrics[f'eval/{data_name}_test'] = float(v[2])
            val_all.append(v[0])
            test_all.append(v[2])
            if data_name in in_domain:
                val_in.append(v[0])
                test_in.append(v[2])
            else:
                val_out.append(v[0])
                test_out.append(v[2])

    metrics['eval/macro_val'] = macro(val_all)
    metrics['eval/macro_test'] = macro(test_all)
    metrics['eval/in_domain_val'] = macro(val_in)
    metrics['eval/in_domain_test'] = macro(test_in)
    metrics['eval/cross_domain_val'] = macro(val_out)
    metrics['eval/cross_domain_test'] = macro(test_out)

    # Equal weight per domain rather than per dataset: with an uneven split
    # (e.g. 5 in-domain vs 3 cross), eval/macro_* silently favours whichever
    # domain has more datasets.
    def balanced(a: float, b: float) -> float:
        parts = [m for m in (a, b) if m == m]  # drop NaN if a side is empty
        return macro(parts)

    metrics['eval/domain_balanced_val'] = balanced(
        metrics['eval/in_domain_val'], metrics['eval/cross_domain_val']
    )
    metrics['eval/domain_balanced_test'] = balanced(
        metrics['eval/in_domain_test'], metrics['eval/cross_domain_test']
    )

    logging.info('--------------------------------')
    logging.info(
        'AVE-ALL-VAL: {:.5f} | AVE-ALL-TEST: {:.5f}'.format(
            metrics['eval/macro_val'], metrics['eval/macro_test']
        )
    )
    logging.info(
        'IN-DOMAIN-VAL: {:.5f} | IN-DOMAIN-TEST: {:.5f} | '
        'CROSS-DOMAIN-VAL: {:.5f} | CROSS-DOMAIN-TEST: {:.5f}'.format(
            metrics['eval/in_domain_val'],
            metrics['eval/in_domain_test'],
            metrics['eval/cross_domain_val'],
            metrics['eval/cross_domain_test'],
        )
    )
    logging.info('--------------------------------')

    if use_wandb:
        wandb.log(metrics, step=step)

    return metrics['eval/macro_val'], metrics['eval/macro_test'], nc_dict


def train(
    dataloader: DataLoader,
    sampler: DistributedSampler,
    model: DistributedDataParallel,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epochs: int,
    rank: int,
    device: torch.device,
    ckpt_path: Path,
    nc_data: dict,
    nc_tasks: dict,
    model_args: TransferArguments,
    in_domain: Sequence[str],
    use_wandb: bool,
) -> None:
    num_batches = len(dataloader)
    init_time = time.time()
    log_step = getattr(model_args, 'log_step', 100)
    eval_every_steps = getattr(model_args, 'eval_every_steps', 10)

    best_val_acc, best_test_acc, best_epoch = 0.0, 0.0, 0
    best_res_dict: dict = {}

    if rank == 0:
        best_val_acc, best_test_acc, best_res_dict = report_downstream(
            model.module,
            nc_data,
            nc_tasks,
            device,
            model_args,
            in_domain,
            'Initial-Evaluate',
            0,
            use_wandb,
        )

    logging.info('------------Training-------------')
    global_step = 0
    for epoch in range(epochs):
        model.train()
        sampler.set_epoch(epoch)
        epoch_loss_mean, loss_count = 0.0, 0
        window: Dict[str, float] = {}
        window_count = 0

        pbar = tqdm(
            dataloader,
            total=num_batches,
            desc=f'Epoch {epoch + 1}',
            disable=(rank != 0),
            smoothing=0.1,
            leave=False,
        )
        for step, data in enumerate(pbar):
            optimizer.zero_grad(set_to_none=True)
            train_loss = model(data)
            train_loss.backward()
            # AFTER backward: the shared trainer clips before, where every
            # grad is still None and the clip is a silent no-op.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_loss_mean += train_loss.item()
            loss_count += 1

            parts = dict(model.module.last_output)
            parts['gnorm'] = float(grad_norm)
            for key, value in parts.items():
                window[key] = window.get(key, 0.0) + value
            window_count += 1

            if rank == 0 and (step + 1) % log_step == 0:
                mean = {k: v / window_count for k, v in window.items()}
                cross = max(mean.get('cross', float('nan')), 1e-12)
                ratio = (
                    model_args.lambda_graph * mean.get('sigreg_graph', 0.0)
                    + model_args.lambda_text * mean.get('sigreg_text', 0.0)
                ) / cross
                erank = model.module.last_erank
                logging.info(
                    '[epoch %d step %d/%d] total=%.4f cross=%.4f sigreg_g=%.4f '
                    'sigreg_t=%.4f ratio=%.3f gnorm=%.3f erank_g=%.1f erank_t=%.1f '
                    'lr=%.2e',
                    epoch + 1,
                    step + 1,
                    num_batches,
                    mean.get('loss', float('nan')),
                    mean.get('cross', float('nan')),
                    mean.get('sigreg_graph', float('nan')),
                    mean.get('sigreg_text', float('nan')),
                    ratio,
                    mean.get('gnorm', float('nan')),
                    erank.get('graph', float('nan')),
                    erank.get('text', float('nan')),
                    scheduler.get_last_lr()[0],
                )
                pbar.set_postfix(
                    {
                        'cross': f'{mean.get("cross", 0):.3f}',
                        'sg_g': f'{mean.get("sigreg_graph", 0):.3f}',
                        'ratio': f'{ratio:.2f}',
                    }
                )
                if use_wandb:
                    wandb.log(
                        {
                            'train/loss': mean.get('loss', float('nan')),
                            'train/cross': mean.get('cross', float('nan')),
                            'train/sigreg_graph': mean.get(
                                'sigreg_graph', float('nan')
                            ),
                            'train/sigreg_text': mean.get('sigreg_text', float('nan')),
                            'train/ratio': ratio,
                            'train/grad_norm': mean.get('gnorm', float('nan')),
                            'train/erank_graph': erank.get('graph', float('nan')),
                            'train/erank_text': erank.get('text', float('nan')),
                            'train/lr': scheduler.get_last_lr()[0],
                            'train/epoch': epoch + 1,
                        },
                        step=global_step,
                    )
                window, window_count = {}, 0

            if (
                rank == 0
                and eval_every_steps > 0
                and global_step % eval_every_steps == 0
            ):
                ave_val, ave_test, nc_dict = report_downstream(
                    model.module,
                    nc_data,
                    nc_tasks,
                    device,
                    model_args,
                    in_domain,
                    f'Evaluate-step-{global_step}',
                    global_step,
                    use_wandb,
                )
                if ave_val > best_val_acc:
                    best_val_acc, best_test_acc = ave_val, ave_test
                    best_res_dict, best_epoch = deepcopy(nc_dict), epoch + 1
                model.train()

        pbar.close()
        if rank == 0:
            logging.info(
                'EPOCH {:05d} | TRAIN LOSS: {:.5f}'.format(
                    epoch + 1, epoch_loss_mean / max(1, loss_count)
                )
            )
            if use_wandb:
                wandb.log(
                    {
                        'train/epoch_loss': epoch_loss_mean / max(1, loss_count),
                        'train/epoch': epoch + 1,
                    },
                    step=global_step,
                )

        if rank == 0 and epoch % model_args.eval_step == 0:
            ave_val, ave_test, nc_dict = report_downstream(
                model.module,
                nc_data,
                nc_tasks,
                device,
                model_args,
                in_domain,
                f'Evaluate-{epoch + 1}',
                global_step,
                use_wandb,
            )
            if ave_val > best_val_acc:
                best_val_acc, best_test_acc = ave_val, ave_test
                best_res_dict, best_epoch = deepcopy(nc_dict), epoch + 1
            logging.info(
                f'BEST-VAL-ACC: {best_val_acc:.5f} | BEST-TEST-ACC: {best_test_acc:.5f} '
                f'| BEST-EPOCH: {best_epoch}'
            )
            logging.info('--------------------------------')

        if rank == 0 and epoch % model_args.save_step == 0:
            torch.save(
                model.module.state_dict(),
                os.path.join(ckpt_path, f'legtjepa-{epoch}.ckpt'),
            )

        if rank == 0:
            elapsed = str(datetime.timedelta(seconds=time.time() - init_time))
            logging.info(f'TOTAL RUNNING TIME: EPOCH-{epoch + 1}: {elapsed}.')

    if rank == 0:
        logging.info('------------Final-Evaluate-------------')
        logging.info(f'-----Best Result from Epoch {best_epoch}-----')
        for data_name in best_res_dict:
            for method_name in best_res_dict[data_name]:
                v = best_res_dict[data_name][method_name]
                logging.info(
                    'DATA: {} | METHOD: {} \nVAL-ACC: {:.5f}±{:.5f} | TEST-ACC: {:.5f}±{:.5f}'.format(
                        data_name, method_name, v[0], v[1], v[2], v[3]
                    )
                )
        logging.info(
            f'BEST-VAL-ACC: {best_val_acc:.5f} | BEST-TEST-ACC: {best_test_acc:.5f}'
        )
        if use_wandb:
            wandb.log(
                {'best/macro_val': best_val_acc, 'best/macro_test': best_test_acc},
                step=global_step,
            )


def get_model(device: torch.device, model_args: TransferArguments) -> torch.nn.Module:
    if model_args.encoder == 'GAT':
        encoder: GraphEncoder = GATNet(
            384,
            model_args.hidden_dim // model_args.n_head,
            model_args.hidden_dim,
            model_args.n_layers,
            feat_drop=model_args.dropout,
            attn_drop=model_args.attn_drop,
            heads=model_args.n_head,
            norm=model_args.norm,
            activation=model_args.activation,
            use_residual=model_args.use_residual,
        )
    elif model_args.encoder == 'GCN':
        encoder = GCNNet(
            384,
            model_args.hidden_dim,
            model_args.hidden_dim,
            model_args.n_layers,
            dropout=model_args.dropout,
            norm=model_args.norm,
            activation=model_args.activation,
            use_residual=model_args.use_residual,
        )
    else:
        raise ValueError(f'Not implemented: {model_args.encoder}.')

    if model_args.task.lower() != 'legtjepa':
        raise ValueError(
            f'This trainer is LeGTJEPA-only; got task={model_args.task}. '
            f'Use the shared transfer trainer for other objectives.'
        )
    return PretrainLeGTJEPA(encoder, device, model_args)


def load_subgraphs(
    data_dir: str, graph_name: str
) -> Tuple[List[dgl.DGLGraph], List[int], List[int]]:
    subgraphs = dgl.load_graphs(os.path.join(data_dir, f'{graph_name}_subgraphs.dgl'))[
        0
    ]
    n_nodes = [sg.num_nodes() for sg in subgraphs]
    n_edges = [sg.num_edges() for sg in subgraphs]
    return subgraphs, n_nodes, n_edges


def pretrain(
    rank: int,
    world_size: int,
    port: int,
    meta_args: MetaArguments,
    model_args: TransferArguments,
    data_args: DataArguments,
    eval_data_dir: Path,
    eval_task_dir: Path,
    ckpt_path: Path,
) -> None:
    setup_logging(meta_args.log_file_path)
    init_process_group(world_size, rank, port)
    if torch.cuda.is_available():
        device = torch.device('cuda:{:d}'.format(rank))
        torch.cuda.set_device(device)
        logging.info(f'Using GPU {device}')
    else:
        device = torch.device('cpu')

    # --- wandb / sweep ---------------------------------------------------
    # mp.spawn starts fresh interpreters, so wandb is initialized here rather
    # than in main(); WANDB_SWEEP_ID is inherited from the agent's env.
    sweep_id = os.environ.get('WANDB_SWEEP_ID')
    use_wandb = bool(getattr(meta_args, 'verbose', False)) or sweep_id is not None
    payload: list = [{}, '']
    if use_wandb and rank == 0:
        mode: Literal['online', 'offline'] = (
            'online'
            if sweep_id is not None
            else ('offline' if getattr(meta_args, 'wandb_offline', True) else 'online')
        )
        wandb.init(
            project=getattr(meta_args, 'wandb_project', 'legtjepa-gstbench'),
            name=getattr(meta_args, 'wandb_run_name', None) or None,
            config={'world_size': world_size, 'global_seed': meta_args.global_seed},
            mode=mode,
        )
        time.sleep(5)  # Helpful for a potential 409 error on wandb servers.
        assert wandb.run is not None
        payload = [dict(wandb.config), wandb.run.id]
    if sweep_id is not None and world_size > 1:
        dist.broadcast_object_list(payload, src=0)
    sweep_overrides, _run_id = payload

    for key, value in sweep_overrides.items():
        if hasattr(model_args, key):
            setattr(model_args, key, value)
        elif rank == 0 and key not in ('world_size', 'global_seed'):
            logging.warning(f'sweep override {key!r} is not a model field; ignored.')
    # ---------------------------------------------------------------------

    seed_everything(meta_args.global_seed)

    dataset: Optional[Union[LazyGraphDataset, List[Any]]] = None
    if data_args.data_name == 'papers100M':
        pretrain_root = get_scratch() / 'GSTBench'
        graph_list, _, _ = load_subgraphs(
            str(pretrain_root / 'subgraphs'), 'papers100M'
        )
        feature_path = str(pretrain_root / 'emb' / 'sbert_embeddings_con_split.npy')
        if data_args.pretrain_data_ids[0] != -1:
            graph_list = [graph_list[i] for i in data_args.pretrain_data_ids]
        elif data_args.pretrain_data_size != -1:
            selected = torch.randperm(len(graph_list))[: data_args.pretrain_data_size]
            graph_list = [graph_list[i] for i in selected]
        dataset = LazyGraphDataset(graph_list, feature_path)
    else:
        try:
            pyg_graph = torch.load(
                os.path.join(eval_data_dir, f'{data_args.data_name}_fixed_sbert.pt')
            )
        except Exception:
            pyg_graph = torch.load(
                os.path.join(eval_data_dir, f'{data_args.data_name}.pt')
            )
        src, dst = pyg_graph.edge_index
        graph = dgl.graph((src, dst), num_nodes=pyg_graph.num_nodes)
        graph.ndata['feat'] = pyg_graph.x
        dataset = [graph]

    assert isinstance(dataset, Dataset)

    collator = Universal_Collator(model_args.task, model_args, device)
    sampler: DistributedSampler = DistributedSampler(
        dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=collator,
        worker_init_fn=worker_init_fn,
        num_workers=model_args.num_workers,
        pin_memory=True,
    )

    model = get_model(device, model_args).to(device)
    model = DistributedDataParallel(
        model,
        device_ids=None if device.type == 'cpu' else [device],
        output_device=None if device.type == 'cpu' else device,
        find_unused_parameters=False,
    )
    logging.info('total params: %d', sum(p.numel() for p in model.parameters()))

    opt_cls = {
        'adamw': torch.optim.AdamW,
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD,
    }[model_args.opt]
    optimizer = opt_cls(
        model.module.trainable_parameters(),
        lr=0.0,
        weight_decay=model_args.weight_decay,
    )

    if model_args.scheduler == 'cosine':
        tot_steps = len(dataloader) * model_args.epochs
        warmup_steps = (
            len(dataloader)
            if model_args.warmup_steps == -1
            else model_args.warmup_steps
        )
        logging.info(f'TOTAL OPTIM STEPS: {tot_steps} | WARMUP STEPS: {warmup_steps}.')
        scheduler: LRScheduler = WarmupCosineDecayScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=tot_steps,
            max_lr=model_args.peak_lr,
        )
    elif model_args.scheduler == 'constant':
        scheduler = ConstantLRScheduler(optimizer, model_args.peak_lr)
    else:
        raise ValueError(f'Unrecognized scheduler: {model_args.scheduler}.')

    nc_data = get_node_data_all(data_args.eval_data_names, str(eval_data_dir))
    nc_tasks = {
        name: create_k_shot_tasks(
            str(eval_task_dir),
            name,
            d['y'],
            model_args.n_tasks,
            model_args.n_shots,
            model_args.n_val,
            model_args.eval_data_seed,
        )
        for name, d in nc_data.items()
    }

    in_domain = tuple(
        getattr(data_args, 'in_domain_data_names', None) or DEFAULT_IN_DOMAIN
    )
    if rank == 0:
        present = [n for n in data_args.eval_data_names if n in in_domain]
        absent = [n for n in data_args.eval_data_names if n not in in_domain]
        logging.info(f'IN-DOMAIN: {present} | CROSS-DOMAIN: {absent}')

    gc.collect()
    seed_everything(meta_args.global_seed)
    try:
        train(
            dataloader,
            sampler,
            model,
            optimizer,
            scheduler,
            model_args.epochs,
            rank,
            device,
            ckpt_path,
            nc_data,
            nc_tasks,
            model_args,
            in_domain,
            use_wandb and rank == 0,
        )
    finally:
        if use_wandb and rank == 0 and wandb.run is not None:
            wandb.finish()
        logging.info('Optimization Finished!')
        dist.destroy_process_group()


def main() -> None:
    root = get_root_dir()
    args = parser.parse_args()
    meta_args, experiment_args = parse_args(root / args.config_file)
    root_dir = Path(str(meta_args.root_dir))
    seed_everything(meta_args.global_seed)
    setup_logging(meta_args.log_file_path)

    eval_data_dir = root_dir / 'downstream_root'
    eval_data_dir.mkdir(parents=True, exist_ok=True)
    eval_task_dir = root_dir / 'eval_task_dir'
    eval_task_dir.mkdir(parents=True, exist_ok=True)

    for experiment, experiment_arg in experiment_args.exp_args.items():
        ckpt_path = root_dir / 'weights' / f'ckpt-{experiment}'
        ckpt_path.mkdir(parents=True, exist_ok=True)
        port = random.randint(10000, 65535)
        mp.spawn(
            pretrain,
            args=(
                args.num_gpus,
                port,
                meta_args,
                experiment_arg.model_args,
                experiment_arg.data_args,
                eval_data_dir,
                eval_task_dir,
                ckpt_path,
            ),
            nprocs=args.num_gpus,
        )


if __name__ == '__main__':
    main()
