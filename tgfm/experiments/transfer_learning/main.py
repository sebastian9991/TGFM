"""Adapted from GSTBench's train_ssl.py.

Changes relative to the original:
    - PretrainSIGReg is registered as a task ('sigreg'); other SSL methods removed.
    - eval_downstream reports linear probing only (Table 3).
    - Configuration via YAML (TransferArguments / DataArguments / MetaArguments),
      logging via tgfm.utils.logger — same conventions as the SSGE trainer.
Everything else (DDP setup, data pipeline, schedulers, checkpointing,
best-checkpoint selection on mean downstream val accuracy) is unchanged.

Original code can be found here: https://github.com/SongYYYY/GSTBench/tree/main.
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
from typing import Any, List, Optional, Tuple, Union

import dgl
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from tgfm.dataset.lazy_graph import LazyGraphDataset
from tgfm.evaluation.transfer_eval import (
    create_k_shot_tasks,
    eval_downstream,
    get_node_data_all,
)
from tgfm.models.base_models.base_models import GATNet, GCNNet
from tgfm.models.pretrain_model.SIGReg import PretrainSIGReg
from tgfm.utils.args import (
    DataArguments,
    MetaArguments,
    TransferArguments,
    parse_args,
)
from tgfm.utils.collate import Universal_Collator
from tgfm.utils.logger import setup_logging
from tgfm.utils.lr import ConstantLRScheduler, WarmupCosineDecayScheduler
from tgfm.utils.path import get_root_dir
from tgfm.utils.seed import seed_everything
from tgfm.utils.transfer_utils import (
    estimate_remaining_time,
    init_process_group,
    worker_init_fn,
)

GraphEncoder = Union[GCNNet, GATNet]

os.environ['DGLBACKEND'] = 'pytorch'


parser = argparse.ArgumentParser(
    description='Transfer Learning',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file', type=str, required=True, help='Path to configuration file.'
)
parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs on node.')


def train(
    dataloader: DataLoader,
    sampler: DistributedSampler,
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    task: str,
    epochs: int,
    rank: int,
    device: torch.device,
    ckpt_path: Path,
    nc_data: dict,
    nc_tasks: dict,
    model_args: TransferArguments,
) -> None:
    num_batches = len(dataloader)
    init_time = time.time()

    if rank == 0:
        logging.info(f'------------Initial-Evaluate-------------')
        nc_dict = eval_downstream(model.module, nc_data, nc_tasks, device, model_args)
        ave_acc_val, ave_acc_test, ave_count = 0.0, 0.0, 0.0
        for data_name in nc_dict.keys():
            for method_name in nc_dict[data_name].keys():
                logging.info(
                    'DATA: {} | METHOD: {} \n'
                    'VAL-ACC: {:.5f}±{:.5f} | TEST-ACC: {:.5f}±{:.5f}'.format(
                        data_name,
                        method_name,
                        nc_dict[data_name][method_name][0],
                        nc_dict[data_name][method_name][1],
                        nc_dict[data_name][method_name][2],
                        nc_dict[data_name][method_name][3],
                    )
                )

                ave_acc_val += nc_dict[data_name][method_name][0]
                ave_acc_test += nc_dict[data_name][method_name][2]
                ave_count += 1

        ave_acc_val /= ave_count
        ave_acc_test /= ave_count
        logging.info('--------------------------------')
        logging.info(
            f'AVE-ALL-VAL: {ave_acc_val:.5f} | AVE-ALL-TEST: {ave_acc_test:.5f}'
        )
        logging.info('--------------------------------')

    # best results
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0.0

    # training
    logging.info(f'------------Training-------------')
    for epoch in range(epochs):
        model.train()
        sampler.set_epoch(epoch)
        epoch_loss_mean = 0
        loss_count = 0
        step = 0
        start_time = time.time()

        for data in dataloader:
            optimizer.zero_grad()
            train_loss = model(data)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            train_loss.backward()
            optimizer.step()
            scheduler.step()

            # record loss
            epoch_loss_mean += train_loss.item()
            loss_count += 1

            if rank == 0:
                step += 1
                estimate_remaining_time(start_time, step, num_batches, 100)

        if rank == 0:
            logging.info(
                'EPOCH {:05d} | TRAIN LOSS: {:.5f}'.format(
                    epoch + 1, epoch_loss_mean / loss_count
                )
            )

        # evaluate per epoch
        if rank == 0 and epoch % model_args.eval_step == 0:
            logging.info(f'------------Evaluate-{epoch + 1}------------')
            nc_dict = eval_downstream(
                model.module, nc_data, nc_tasks, device, model_args
            )
            ave_acc_val, ave_acc_test, ave_count = 0.0, 0.0, 0.0
            for data_name in nc_dict.keys():
                for method_name in nc_dict[data_name].keys():
                    logging.info(
                        'DATA: {} | METHOD: {} \n'
                        'VAL-ACC: {:.5f}±{:.5f} | TEST-ACC: {:.5f}±{:.5f}'.format(
                            data_name,
                            method_name,
                            nc_dict[data_name][method_name][0],
                            nc_dict[data_name][method_name][1],
                            nc_dict[data_name][method_name][2],
                            nc_dict[data_name][method_name][3],
                        )
                    )

                    ave_acc_val += nc_dict[data_name][method_name][0]
                    ave_acc_test += nc_dict[data_name][method_name][2]
                    ave_count += 1

            ave_acc_val /= ave_count
            ave_acc_test /= ave_count
            logging.info('--------------------------------')
            logging.info(
                f'AVE-ALL-VAL: {ave_acc_val:.5f} | AVE-ALL-TEST: {ave_acc_test:.5f}'
            )
            logging.info('--------------------------------')
            if ave_acc_val > best_val_acc:
                best_val_acc = ave_acc_val
                best_test_acc = ave_acc_test
                best_res_dict = deepcopy(nc_dict)
                best_epoch = epoch + 1
            logging.info(
                f'BEST-VAL-ACC: {best_val_acc:.5f} | BEST-TEST-ACC: {best_test_acc:.5f} | BEST-EPOCH: {best_epoch}'
            )
            logging.info('--------------------------------')

        if rank == 0 and epoch % model_args.save_step == 0:
            torch.save(
                model.module.state_dict(),
                os.path.join(ckpt_path, f'{task}-{epoch}.ckpt'),
            )

        if rank == 0:
            epoch_time = time.time()
            running_time = epoch_time - init_time
            formatted_time = str(datetime.timedelta(seconds=running_time))
            logging.info(
                'TOTAL RUNNING TIME: EPOCH-{}: {}.'.format(epoch + 1, formatted_time)
            )
            logging.info('--------------------------------')

    if rank == 0:
        logging.info(f'------------Final-Evaluate-------------')
        logging.info(f'-----Best Result from Epoch {best_epoch}-----')
        for data_name in best_res_dict.keys():
            for method_name in best_res_dict[data_name].keys():
                logging.info(
                    'DATA: {} | METHOD: {} \n'
                    'VAL-ACC: {:.5f}±{:.5f} | TEST-ACC: {:.5f}±{:.5f}'.format(
                        data_name,
                        method_name,
                        best_res_dict[data_name][method_name][0],
                        best_res_dict[data_name][method_name][1],
                        best_res_dict[data_name][method_name][2],
                        best_res_dict[data_name][method_name][3],
                    )
                )
        logging.info('--------------------------------')
        logging.info(
            f'BEST-VAL-ACC: {best_val_acc:.5f} | BEST-TEST-ACC: {best_test_acc:.5f}'
        )
        logging.info('--------------------------------')


def get_model(device: torch.device, model_args: TransferArguments) -> torch.nn.Module:
    encoder: Optional[GraphEncoder] = None
    if model_args.encoder == 'GAT':
        hidden_dim = model_args.hidden_dim // model_args.n_head
        encoder = GATNet(
            384,
            hidden_dim,
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

    if model_args.task.lower() == 'sigreg':
        pretrain_model = PretrainSIGReg(encoder, device, model_args)
    else:
        raise ValueError(f'Not implemented: {model_args.task}.')

    return pretrain_model


def load_subgraphs(
    data_dir: str, graph_name: str
) -> Tuple[List[dgl.DGLGraph], List[int], List[int]]:
    n_node_list = []
    n_edge_list = []
    subgraphs = dgl.load_graphs(os.path.join(data_dir, f'{graph_name}_subgraphs.dgl'))[
        0
    ]
    for i in range(len(subgraphs)):
        sg = subgraphs[i]
        n_node_list.append(sg.num_nodes())
        n_edge_list.append(sg.num_edges())

    return subgraphs, n_node_list, n_edge_list


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
    # Spawned processes start fresh interpreters: logging and seeds must be
    # re-initialized per rank.
    setup_logging(meta_args.log_file_path)
    init_process_group(world_size, rank, port)
    if torch.cuda.is_available():
        device = torch.device('cuda:{:d}'.format(rank))
        torch.cuda.set_device(device)
        logging.info(f'Using GPU {device}')
    else:
        device = torch.device('cpu')

    seed_everything(meta_args.global_seed)

    # prepare pretraining data
    dataset: Optional[Union[LazyGraphDataset, List[Any]]] = None
    # TODO: dyanmic folder locations
    if 'papers100M' == data_args.data_name:
        data_dir = '$SCRATCH/GSTBench/subgraph/'
        graph_name = 'papers100M'
        feature_path = '$SCRATCH/GSTBench/emb/sbert_embeddings_con_split.npy'
        graph_list, n_node_list, n_edge_list = load_subgraphs(data_dir, graph_name)
        if data_args.pretrain_data_ids[0] != -1:
            graph_list = [graph_list[id] for id in data_args.pretrain_data_ids]
        elif data_args.pretrain_data_size != -1:
            selected_ids = torch.randperm(len(graph_list))[
                : data_args.pretrain_data_size
            ]
            graph_list = [graph_list[id] for id in selected_ids]
        dataset = LazyGraphDataset(graph_list, feature_path)
    else:
        try:
            pyg_graph = torch.load(
                os.path.join(eval_data_dir, f'{data_args.data_name}_fixed_sbert.pt')
            )
        except:
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

    if device.type == 'cpu':
        model = DistributedDataParallel(model, find_unused_parameters=False)
    else:
        model = DistributedDataParallel(
            model,
            device_ids=[device],
            output_device=device,
            find_unused_parameters=False,
        )

    logging.info(model)
    logging.info('total params: %d', sum(p.numel() for p in model.parameters()))

    optimizer: Optional[Optimizer] = None
    if model_args.opt == 'adamw':
        optimizer = torch.optim.AdamW(
            model.module.trainable_parameters(),
            lr=0.0,
            weight_decay=model_args.weight_decay,
        )
    elif model_args.opt == 'adam':
        optimizer = torch.optim.Adam(
            model.module.trainable_parameters(),
            lr=0.0,
            weight_decay=model_args.weight_decay,
        )
    elif model_args.opt == 'sgd':
        optimizer = torch.optim.SGD(
            model.module.trainable_parameters(),
            lr=0.0,
            weight_decay=model_args.weight_decay,
        )

    scheduler: Optional[LRScheduler] = None
    assert optimizer is not None
    if model_args.scheduler == 'cosine':
        tot_steps = len(dataloader) * model_args.epochs
        warmup_steps = (
            len(dataloader)
            if model_args.warmup_steps == -1
            else model_args.warmup_steps
        )
        logging.info(f'TOTAL OPTIM STEPS: {tot_steps} | WARMUP STEPS: {warmup_steps}.')
        scheduler = WarmupCosineDecayScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=tot_steps,
            max_lr=model_args.peak_lr,
        )
    elif model_args.scheduler == 'constant':
        scheduler = ConstantLRScheduler(optimizer, model_args.peak_lr)
    else:
        raise ValueError(f'Unrecognized scheduler: {model_args.scheduler}.')

    # downstream data
    nc_data = get_node_data_all(data_args.eval_data_names, str(eval_data_dir))
    nc_tasks = {}
    for data_name, data in nc_data.items():
        nc_tasks[data_name] = create_k_shot_tasks(
            str(eval_task_dir),
            data_name,
            data['y'],
            model_args.n_tasks,
            model_args.n_shots,
            model_args.n_val,
            model_args.eval_data_seed,
        )

    gc.collect()

    seed_everything(meta_args.global_seed)
    train(
        dataloader,
        sampler,
        model,
        optimizer,
        scheduler,
        model_args.task,
        model_args.epochs,
        rank,
        device,
        ckpt_path,
        nc_data,
        nc_tasks,
        model_args,
    )

    logging.info('Optimization Finished!')
    logging.info('--')
    dist.destroy_process_group()


def main() -> None:
    root = get_root_dir()
    args = parser.parse_args()
    config_file_path = root / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)
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

        # The port is chosen once in the parent so every spawned rank joins the
        # same process group (per-rank randint would desynchronize and hang).
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
