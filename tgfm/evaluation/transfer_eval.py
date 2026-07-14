"""Evaluation helpers for the Table 3 (linear probing) experiment.

All functions are taken from GSTBench's eval_helper.py, trimmed to the linear
probing path only. eval_downstream now reports LINEAR results exclusively.
"""

import os
import pickle
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tgfm.utils.args import ModelArguments, TransferArguments
from tgfm.utils.seed import seed_everything


def get_node_data_all(
    data_names: List[str], data_dir: str
) -> Dict[str, Dict[str, Tensor]]:
    node_data_all: Dict = defaultdict(dict)
    for data_name in data_names:
        try:
            data = torch.load(os.path.join(data_dir, f'{data_name}_fixed_sbert.pt'))
        except:
            try:
                data = torch.load(os.path.join(data_dir, f'{data_name}.pt'))
            except:
                raise ValueError(f'file does not exist: {data_name}')

        node_data_all[data_name]['x'] = data.x
        node_data_all[data_name]['y'] = data.y
        node_data_all[data_name]['edge_index'] = data.edge_index

    return node_data_all


def save_tasks(tasks: List[Dict[str, Tensor]], file_path: str) -> None:
    with open(file_path, 'wb') as f:
        pickle.dump(tasks, f)


def load_tasks(file_path: str) -> List[Dict[str, Tensor]]:
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def create_k_shot_tasks(
    data_root: str,
    data_name: str,
    labels: Tensor,
    M: int,
    K: int,
    n_val: int,
    seed: int = 0,
) -> List[Dict[str, Tensor]]:
    """Create or load M K-shot tasks.

    Args:
        data_root (str): Directory where the tasks should be saved/loaded from.
        data_name (str): A name to identify the dataset, used for saving/loading tasks.
        labels (torch.Tensor or array-like): The label for each sample in the dataset.
        M (int): Number of tasks.
        K (int): Number of examples per class in the training set (K-shot).
        n_val (int): Total number of examples (across all classes) in the validation set.
        seed (int): Random seed for reproducibility.

    Returns:
        list of dicts: Each dict contains three boolean masks (torch.BoolTensor):
            {
                'train_mask': torch.BoolTensor,
                'val_mask':   torch.BoolTensor,
                'test_mask':  torch.BoolTensor
            }
            The masks have the same length as `labels`.
    """
    # If labels is not already a torch.Tensor, convert it:
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long)

    # Construct the filename for saving/loading
    file_name = f'{data_name}_{K}_{n_val}_{M}_{seed}.pkl'
    file_path = os.path.join(data_root, file_name)

    # If a task file already exists, load and return it
    if os.path.exists(file_path):
        return load_tasks(file_path)

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Unique labels and number of samples
    unique_labels = torch.unique(labels)
    num_samples = len(labels)  # total number of samples in the dataset

    tasks = []

    # Create M tasks
    for _ in range(M):
        # Create boolean masks (torch) for train, val, and test
        train_mask = torch.zeros(num_samples, dtype=torch.bool)
        val_mask = torch.zeros(num_samples, dtype=torch.bool)
        test_mask = torch.zeros(num_samples, dtype=torch.bool)

        # For each class, pick which samples go to train
        for c in unique_labels:
            # Indices for this class
            class_indices = torch.where(labels == c)[0]

            # Shuffle
            shuffled_indices = class_indices[torch.randperm(len(class_indices))]

            # Number of samples we have in this class
            n_class_samples = len(shuffled_indices)

            # Train indices (K per class)
            n_train = min(K, n_class_samples)
            train_indices = shuffled_indices[:n_train]

            train_mask[train_indices] = True

        # The remaining indices are potential val/test candidates
        remaining_indices = torch.nonzero(~train_mask, as_tuple=True)[0]
        shuffled_remaining = remaining_indices[torch.randperm(len(remaining_indices))]

        # Pick n_val from the remaining as validation
        n_val_actual = min(n_val, len(shuffled_remaining))
        val_indices = shuffled_remaining[:n_val_actual]
        test_indices = shuffled_remaining[n_val_actual:]

        val_mask[val_indices] = True
        test_mask[test_indices] = True

        task_dict = {
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask,
        }
        tasks.append(task_dict)

    # Save the tasks to disk for future reuse
    with open(file_path, 'wb') as f:
        pickle.dump(tasks, f)

    return tasks


# Embedding Eval: Linear probing
def get_linear_results(
    node_embs: Tensor,
    data: Dict[str, Tensor],
    train_mask: Tensor,
    val_mask: Tensor,
    test_mask: Tensor,
    lr: float,
    l2: float,
    dropout: float,
    args: ModelArguments,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    assert isinstance(ModelArguments, TransferArguments)
    data['x']
    labels = data['y']
    data['edge_index']
    n_classes = int(labels.max().item() + 1)

    val_acc_list, test_acc_list = [], []
    for i in range(args.linear_runs):
        seed_everything(i)
        clf = torch.nn.Linear(node_embs.shape[1], n_classes).to(device)

        val_acc, test_acc = train_clf(
            clf,
            node_embs,
            labels,
            train_mask,
            val_mask,
            test_mask,
            lr,
            l2,
            dropout,
            device,
        )

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)

    return np.mean(val_acc_list), np.mean(test_acc_list)


def train_clf(
    clf: nn.Module,
    node_embs: Tensor,
    labels: Tensor,
    train_mask: Tensor,
    val_mask: Tensor,
    test_mask: Tensor,
    lr: float = 0.01,
    l2: float = 0,
    dropout: float = 0.2,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=l2)
    best_acc: float = 0.0
    for e in range(300):
        clf.train()
        optimizer.zero_grad()
        x = F.dropout(node_embs[train_mask].to(device), p=dropout, training=True)
        out = clf(x)
        loss = F.cross_entropy(out, labels[train_mask].to(device))
        loss.backward()
        optimizer.step()
        val_acc, val_loss = evaluate_clf(clf, node_embs, labels, val_mask, device)
        if val_acc > best_acc:
            best_acc = val_acc
            weights = deepcopy(clf.state_dict())

    clf.load_state_dict(weights)
    test_acc, test_loss = evaluate_clf(clf, node_embs, labels, test_mask, device)

    return best_acc, test_acc


@torch.no_grad()
def evaluate_clf(
    clf: nn.Module,
    node_embs: Tensor,
    labels: Tensor,
    mask: Tensor,
    device: Optional[torch.device],
) -> Tuple[float, float]:
    clf.eval()
    out = clf(node_embs[mask].to(device))
    pred = out.argmax(dim=1)
    correct = pred.eq(labels[mask].to(device)).sum().item()
    acc = correct / mask.sum().item()
    loss = F.cross_entropy(out, labels[mask].to(device)).item()
    return acc, loss


def get_mean_linear_results(
    pretrain_model: nn.Module,
    data: Dict[str, Tensor],
    tasks: List[Dict[str, Tensor]],
    args: ModelArguments,
    device: Optional[torch.device] = None,
) -> Tuple[float, float, float, float]:
    assert isinstance(ModelArguments, TransferArguments)
    val_acc_list = []
    test_acc_list = []
    lr, l2, dropout = args.linear_lr, args.linear_l2, args.linear_dropout

    x = data['x']
    edge_index = data['edge_index']
    node_embs = pretrain_model.inference(x.to(device), edge_index.t().to(device))
    for task in tasks:
        train_mask, val_mask, test_mask = (
            task['train_mask'],
            task['val_mask'],
            task['test_mask'],
        )
        val_acc, test_acc = get_linear_results(
            node_embs,
            data,
            train_mask,
            val_mask,
            test_mask,
            lr,
            l2,
            dropout,
            args,
            device,
        )
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)

    return (
        np.mean(val_acc_list),
        np.std(val_acc_list),
        np.mean(test_acc_list),
        np.std(test_acc_list),
    )


def eval_downstream(
    pretrain_model: nn.Module,
    all_data: Dict[str, Dict[str, Tensor]],
    all_tasks: Dict[str, List[Dict[str, Tensor]]],
    device: torch.device,
    args: ModelArguments,
) -> Dict[str, Dict[str, List[float]]]:
    """Linear probing only (Table 3 in the GSTBench paper)."""
    res_dict: Dict = defaultdict(dict)
    for data_name, data in all_data.items():
        val_acc_mean, val_acc_std, test_acc_mean, test_acc_std = (
            get_mean_linear_results(
                pretrain_model, data, all_tasks[data_name], args, device
            )
        )
        res_dict[data_name]['LINEAR'] = [
            val_acc_mean,
            val_acc_std,
            test_acc_mean,
            test_acc_std,
            0,
            0,
        ]

    return res_dict
