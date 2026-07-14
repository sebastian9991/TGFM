"""Faithful PyG port of SSGE's evaluation protocol.

This is optional. The training script (`ssge_main.py`) defaults to your
existing `evaluate_linear_probe` / `node_classification` so SSGE is scored on
*your* pipeline. Use this module instead when you want bit-for-bit comparison
against the numbers in the SSGE paper, which come from a specific logistic-
regression probe (3000 epochs, per-dataset lr2/wd2, public splits for
Cora/CiteSeer/PubMed/WikiCS, random 10/80 split otherwise) plus K-means
clustering.

Differences from the reference: this uses PyG tensors / masks rather than DGL;
the math (probe architecture, training schedule, model selection on val micro-F1,
K-means with Hungarian matching) is unchanged. Clustering needs the ``munkres``
package; if it is missing, ``node_clustering`` is skipped with a warning.
"""

import logging
import random
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
)
from torch import Tensor


def _fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_random(
    n_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8
) -> dict:
    """Random split used by the paper for datasets without public masks."""
    train_size = int(n_samples * train_ratio)
    test_size = int(n_samples * test_ratio)
    idx = torch.randperm(n_samples)
    return {
        'train': idx[:train_size],
        'valid': idx[train_size : train_size + test_size],
        'test': idx[train_size + test_size :],
    }


class _LogReg(torch.nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


def _evaluate_one(
    x: Tensor,
    y: Tensor,
    split: dict,
    num_epochs: int = 3000,
    lr: float = 0.01,
    wd: float = 0.0,
    test_interval: int = 20,
) -> dict:
    device = x.device
    x = x.detach()
    y = y.to(device)
    clf = _LogReg(x.size(1), int(y.max().item()) + 1).to(device)
    optim = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    criterion = torch.nn.NLLLoss()

    best_val, best_mi, best_ma = 0.0, 0.0, 0.0
    for epoch in range(num_epochs):
        clf.train()
        optim.zero_grad()
        out = clf(x[split['train']])
        loss = criterion(log_softmax(out), y[split['train']])
        loss.backward()
        optim.step()

        if (epoch + 1) % test_interval == 0:
            clf.eval()
            with torch.no_grad():
                y_test = y[split['test']].cpu().numpy()
                pred_test = clf(x[split['test']]).argmax(-1).cpu().numpy()
                mi = f1_score(y_test, pred_test, average='micro')
                ma = f1_score(y_test, pred_test, average='macro')

                y_val = y[split['valid']].cpu().numpy()
                pred_val = clf(x[split['valid']]).argmax(-1).cpu().numpy()
                val_mi = f1_score(y_val, pred_val, average='micro')
            if val_mi > best_val:
                best_val, best_mi, best_ma = val_mi, mi, ma
    return {'MiF1': best_mi, 'MaF1': best_ma}


def node_classification(
    z: Tensor,
    y: Tensor,
    dataset: str,
    masks: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
    n_repeats: int = 10,
    lr: float = 0.01,
    wd: float = 0.0,
) -> dict:
    """LR probe matching the SSGE protocol. Returns mean/std micro & macro F1."""
    _fix_seed(0)
    n = z.shape[0]
    mi_list, ma_list = [], []

    if dataset == 'WikiCS':
        # WikiCS ships 20 train/val splits and a single test mask.
        assert masks is not None
        train_masks, val_masks, test_mask = masks
        idx = torch.arange(n, device=z.device)
        for i in range(20):
            split = {
                'train': idx[train_masks[:, i]],
                'valid': idx[val_masks[:, i]],
                'test': idx[test_mask],
            }
            res = _evaluate_one(z, y, split, num_epochs=3000, lr=lr, wd=wd)
            mi_list.append(res['MiF1'])
            ma_list.append(res['MaF1'])
    elif masks is not None:
        train_mask, val_mask, test_mask = masks
        idx = torch.arange(n, device=z.device)
        split = {
            'train': idx[train_mask],
            'valid': idx[val_mask],
            'test': idx[test_mask],
        }
        for _ in range(n_repeats):
            res = _evaluate_one(z, y, split, num_epochs=3000, lr=lr, wd=wd)
            mi_list.append(res['MiF1'])
            ma_list.append(res['MaF1'])
    else:
        for _ in range(n_repeats):
            split = _split_random(n, train_ratio=0.1, test_ratio=0.8)
            res = _evaluate_one(z, y, split, num_epochs=3000, lr=lr, wd=wd)
            mi_list.append(res['MiF1'])
            ma_list.append(res['MaF1'])

    mi = np.array(mi_list) * 100
    ma = np.array(ma_list) * 100
    out = {
        'MiF1_mean': mi.mean(),
        'MiF1_std': mi.std(),
        'MaF1_mean': ma.mean(),
        'MaF1_std': ma.std(),
    }
    logging.info(
        'MiF1=%.2f+-%.2f, MaF1=%.2f+-%.2f',
        out['MiF1_mean'],
        out['MiF1_std'],
        out['MaF1_mean'],
        out['MaF1_std'],
    )
    return out


def _cluster_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Optional[Tuple[float, float, float, float]]:
    """ACC (Hungarian-matched), NMI, ARI, macro-F1."""
    from munkres import Munkres  # local import: optional dependency

    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    ari = adjusted_rand_score(y_true, y_pred)

    y_true = y_true - y_true.min()
    l1 = list(set(y_true))
    l2 = list(set(y_pred))
    if len(l1) != len(l2):
        return None  # cluster count mismatch; skip
    cost = np.zeros((len(l1), len(l2)), dtype=int)
    for i, c1 in enumerate(l1):
        members = [k for k, e in enumerate(y_true) if e == c1]
        for j, c2 in enumerate(l2):
            cost[i][j] = sum(1 for k in members if y_pred[k] == c2)
    indexes = Munkres().compute((-cost).tolist())
    new_pred = np.zeros(len(y_pred))
    for i, _ in enumerate(l1):
        c2 = l2[indexes[i][1]]
        new_pred[[k for k, e in enumerate(y_pred) if e == c2]] = l1[i]
    acc = accuracy_score(y_true, new_pred)
    f1 = f1_score(y_true, new_pred, average='macro')
    return acc, float(nmi), ari, f1


def node_clustering(
    z: Tensor, y: Tensor, normalize: bool = True, n_repeats: int = 10
) -> dict:
    """K-means clustering eval. Skipped (warning) if ``munkres`` is unavailable."""
    try:
        from munkres import Munkres  # noqa: F401
    except ImportError:
        logging.warning('munkres not installed; skipping node_clustering.')
        return {}

    from sklearn.cluster import KMeans

    if normalize:
        z = torch.nn.functional.normalize(z, p=2, dim=1)
    z_np = z.cpu().numpy()
    y_np = y.cpu().numpy()
    k = int(np.unique(y_np).shape[0])

    accs, nmis, aris, f1s = [], [], [], []
    for i in range(n_repeats):
        _fix_seed(i)
        pred = KMeans(n_clusters=k, random_state=i, n_init=10).fit_predict(z_np)
        m = _cluster_metrics(y_np.copy(), pred)
        if m is None:
            continue
        acc, nmi, ari, f1 = m
        accs.append(acc)
        nmis.append(nmi)
        aris.append(ari)
        f1s.append(f1)

    accs_np, nmis_np, aris_np, f1s_np = map(
        lambda a: np.array(a) * 100, (accs, nmis, aris, f1s)
    )
    assert isinstance(accs_np, np.ndarray)
    assert isinstance(nmis_np, np.ndarray)
    assert isinstance(aris_np, np.ndarray)
    assert isinstance(f1s_np, np.ndarray)
    out = {
        'ACC_mean': accs_np.mean(),
        'ACC_std': accs_np.std(),
        'NMI_mean': nmis_np.mean(),
        'NMI_std': nmis_np.std(),
        'ARI_mean': aris_np.mean(),
        'ARI_std': aris_np.std(),
        'F1_mean': f1s_np.mean(),
        'F1_std': f1s_np.std(),
    }
    logging.info(
        'ACC=%.2f+-%.2f, NMI=%.2f+-%.2f, ARI=%.2f+-%.2f, F1=%.2f+-%.2f',
        out['ACC_mean'],
        out['ACC_std'],
        out['NMI_mean'],
        out['NMI_std'],
        out['ARI_mean'],
        out['ARI_std'],
        out['F1_mean'],
        out['F1_std'],
    )
    return out
