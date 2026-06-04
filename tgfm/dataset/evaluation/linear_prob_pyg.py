"""Linear-probe evaluation for SSL graph encoders.

Two probe variants from BGRL (Thakoor et al., 2023):
    - fit_logistic_regression:               random train/test for datasets
                                              without canonical splits
                                              (Amazon-{Photo,Computers},
                                              Coauthor-{CS,Physics}).
    - fit_logistic_regression_preset_splits: uses dataset-provided masks
                                              (WikiCS's 20 splits, Planetoid).

Plus the encoder-side helper:
    - compute_node_embeddings: runs the SSL encoder over the full graph (or
                               a single large ego-net covering it) and returns
                               (N, dim) node embeddings for the probe.
"""

from typing import Optional

import numpy as np
import torch
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    ShuffleSplit,
    train_test_split,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize
from torch import Tensor
from torch_geometric.data import Data


@torch.no_grad()
def compute_node_embeddings(
    encoder: torch.nn.Module,
    full_data: Data,
    rwse: Tensor,
    device: int,
    se: Optional[Tensor] = None,
) -> Tensor:
    """Run the encoder once over the full graph; return (N, dim) node embeddings.

    For SSL evaluation we want one embedding per node. The pretraining-time
    `build_views` pipeline pools to per-view vectors; this helper instead runs
    the encoder on the entire graph as a single forward pass and returns the
    node-level outputs directly.

    Args:
        encoder:   the trained GraphGPS encoder (set to eval mode here).
        full_data: full-graph PyG Data with .x and .edge_index.
        device: Which  device to use.
        rwse:      (N, K) RWSE precomputed on the full graph.
        se:        optional (N, S) structural encoding, if the encoder has
                   an SE branch.

    Returns:
        (N, dim) Tensor on CPU (float).
    """
    encoder.eval()
    x = full_data.x.to(device)
    edge_index = full_data.edge_index.to(device)
    pe = rwse.to(device) if rwse is not None else None
    se_dev = se.to(device) if se is not None else None
    # Single-graph batch: all nodes belong to graph 0.
    batch = torch.zeros(full_data.num_nodes, dtype=torch.long, device=device)

    h = encoder(
        x=x,
        pe=pe,
        se=se_dev,
        edge_index=edge_index,
        batch=batch,
    )  # (N, dim)
    encoder.train()  # restore training mode for the caller
    return h.detach().cpu()


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    data_random_seed: int = 1,
    repeat: int = 1,
) -> list[float]:
    """Random 20/80 train/test linear probe with grid search over C.

    Mirrors BGRL's `fit_logistic_regression`. Repeating multiple times with
    different random splits gives an estimate of variance over the data
    partition. The `data_random_seed` makes the *sequence* of splits
    reproducible across runs (so split #k is the same split #k every run).
    """
    one_hot_encoder = OneHotEncoder(categories='auto', sparse_output=False)
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(bool)
    X = normalize(X, norm='l2')

    rng = np.random.RandomState(data_random_seed)
    accuracies: list[float] = []
    for _ in range(repeat):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.8,
            random_state=rng,
        )
        logreg = LogisticRegression(solver='liblinear')
        c = 2.0 ** np.arange(-10, 11)
        cv = ShuffleSplit(n_splits=5, test_size=0.5)
        clf = GridSearchCV(
            estimator=OneVsRestClassifier(logreg),
            param_grid=dict(estimator__C=c),
            n_jobs=5,
            cv=cv,
            verbose=0,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)
        accuracies.append(float(metrics.accuracy_score(y_test, y_pred)))
    return accuracies


def fit_logistic_regression_preset_splits(
    X: np.ndarray,
    y: np.ndarray,
    train_masks: np.ndarray,  # (N, num_splits) or (N,) (will be promoted)
    val_masks: np.ndarray,  # same shape as train_masks
    test_mask: np.ndarray,  # (N,)
) -> list[float]:
    """Preset-split linear probe with val-based model selection.

    For each split: grid-search C on train, pick C by val accuracy, then
    report test accuracy at that best C. Average across splits is the
    headline metric.
    """
    one_hot_encoder = OneHotEncoder(categories='auto', sparse_output=False)
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(bool)
    X = normalize(X, norm='l2')

    # Promote 1-D masks to (N, 1) so the loop is uniform.
    if train_masks.ndim == 1:
        train_masks = train_masks[:, None]
        val_masks = val_masks[:, None]

    accuracies: list[float] = []
    for split_id in range(train_masks.shape[1]):
        train_mask, val_mask = train_masks[:, split_id], val_masks[:, split_id]
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        best_test_acc, best_val_acc = 0.0, 0.0
        for c in 2.0 ** np.arange(-10, 11):
            clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c))
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_val)
            y_pred = np.argmax(y_pred, axis=1)
            y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)
            val_acc = metrics.accuracy_score(y_val, y_pred)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                y_pred = clf.predict_proba(X_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)
                best_test_acc = metrics.accuracy_score(y_test, y_pred)
        accuracies.append(float(best_test_acc))
    return accuracies


def evaluate_linear_probe(
    embeddings: Tensor,
    full_data: Data,
    repeat: int = 1,
    data_random_seed: int = 1,
) -> dict:
    """Run the appropriate probe based on what masks the Data object carries.

    Returns a dict with:
        - "accuracies":   list[float] per split / per repeat
        - "mean":         float
        - "std":          float
        - "probe_type":   "preset_splits" | "random"
    """
    X = embeddings.cpu().numpy()
    y = full_data.y.cpu().numpy()

    has_masks = (
        hasattr(full_data, 'train_mask')
        and hasattr(full_data, 'val_mask')
        and hasattr(full_data, 'test_mask')
        and full_data.train_mask is not None
    )

    if has_masks:
        train_masks = full_data.train_mask.cpu().numpy()
        val_masks = full_data.val_mask.cpu().numpy()
        test_mask = full_data.test_mask.cpu().numpy()
        # WikiCS test_mask is 1-D even though train/val are 2-D; preserve that.
        accs = fit_logistic_regression_preset_splits(
            X,
            y,
            train_masks,
            val_masks,
            test_mask,
        )
        probe_type = 'preset_splits'
    else:
        accs = fit_logistic_regression(
            X,
            y,
            data_random_seed=data_random_seed,
            repeat=repeat,
        )
        probe_type = 'random'

    arr = np.array(accs)
    return {
        'accuracies': accs,
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'probe_type': probe_type,
    }
