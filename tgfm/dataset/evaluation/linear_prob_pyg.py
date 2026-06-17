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

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    GridSearchCV,
    ShuffleSplit,
    train_test_split,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize
from torch import Tensor
from torch_geometric.data import Data
from tqdm import tqdm


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
    # TODO: Why would we need to normalize it? Taken from TGRL paper.
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

    assert X.shape[0] == y.shape[0] or has_masks
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


# --------SSGE Evaluation Utilities --------


class LogisticRegressionCustom(torch.nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super(LogisticRegressionCustom, self).__init__()
        self.fc = torch.nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, X: Tensor) -> Tensor:
        Z = self.fc(X)
        return Z


def split4NC(n_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8) -> Dict:
    """Split node set for Node Classification."""
    assert train_ratio + test_ratio < 1
    train_size = int(n_samples * train_ratio)
    test_size = int(n_samples * test_ratio)
    indices = torch.randperm(n_samples)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size : test_size + train_size],
        'test': indices[test_size + train_size :],
    }


class LREvaluator4NC:
    def __init__(
        self,
        num_epochs: int = 5000,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        test_interval: int = 20,
    ):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    def evaluate(self, x: Tensor, y: Tensor, split: dict) -> Dict:
        for key in ['train', 'test', 'valid']:
            assert key in split
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogisticRegressionCustom(input_dim, int(num_classes)).to(device)
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        output_fn = torch.nn.LogSoftmax(dim=-1)
        criterion = torch.nn.NLLLoss()

        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 0

        with tqdm(
            total=self.num_epochs,
            desc='(LR)',
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]',
        ) as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()

                output = classifier(x[split['train']])
                loss = criterion(output_fn(output), y[split['train']])

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    y_test = y[split['test']].detach().cpu().numpy()
                    Y_pred = (
                        classifier(x[split['test']]).argmax(-1).detach().cpu().numpy()
                    )
                    test_micro = f1_score(y_test, Y_pred, average='micro')
                    test_macro = f1_score(y_test, Y_pred, average='macro')

                    y_val = y[split['valid']].detach().cpu().numpy()
                    Y_pred = (
                        classifier(x[split['valid']]).argmax(-1).detach().cpu().numpy()
                    )
                    val_micro = f1_score(y_val, Y_pred, average='micro')

                    if val_micro > best_val_micro:
                        best_val_micro = val_micro
                        best_test_micro = test_micro
                        best_test_macro = test_macro

                    pbar.set_postfix(
                        {'best test MiF1': best_test_micro, 'MaF1': best_test_macro}
                    )
                    pbar.update(self.test_interval)

        return {'MiF1': best_test_micro, 'MaF1': best_test_macro}


def node_classification(
    Z: Tensor,
    Y: Tensor,
    dataset: str,
    n_repeats: int = 10,
    lr: float = 0.01,
    wd: float = 1e-4,
    masks: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
) -> None:
    """Evaluate node representations on node classification."""
    # fix_seed(0)
    n_nodes = Z.shape[0]
    MiF1s: List[Any] = []
    MaF1s: List[Any] = []
    if dataset == 'WikiCS':
        assert masks is not None
        train_masks = masks[0]
        val_masks = masks[1]
        test_mask = masks[2]
        indices = torch.arange(n_nodes, device=Z.device)
        for i in range(20):
            split = {
                'train': indices[train_masks[:, i]],
                'valid': indices[val_masks[:, i]],
                'test': indices[test_mask],
            }
            res = LREvaluator4NC(
                num_epochs=3000, learning_rate=lr, weight_decay=wd
            ).evaluate(Z, Y, split)
            MiF1s.append(res['MiF1'])
            MaF1s.append(res['MaF1'])
    else:
        if masks is not None:
            train_mask = masks[0]
            val_mask = masks[1]
            test_mask = masks[2]
            indices = torch.arange(n_nodes, device=Z.device)
            for i in range(n_repeats):
                split = {
                    'train': indices[train_mask],
                    'valid': indices[val_mask],
                    'test': indices[test_mask],
                }
                res = LREvaluator4NC(
                    num_epochs=3000, learning_rate=lr, weight_decay=wd
                ).evaluate(Z, Y, split)
                MiF1s.append(res['MiF1'])
                MaF1s.append(res['MaF1'])
        else:
            for i in range(n_repeats):
                split = split4NC(n_nodes, train_ratio=0.1, test_ratio=0.8)
                res = LREvaluator4NC(
                    num_epochs=3000, learning_rate=lr, weight_decay=wd
                ).evaluate(Z, Y, split)
                MiF1s.append(res['MiF1'])
                MaF1s.append(res['MaF1'])
    MiF1s_np = np.array(MiF1s)
    assert isinstance(MiF1s_np, np.ndarray)
    MaF1s_np = np.array(MaF1s)
    assert isinstance(MaF1s_np, np.ndarray)
    micro_mean = MiF1s_np.mean() * 100
    micro_std = MiF1s_np.std() * 100
    macro_mean = MaF1s_np.mean() * 100
    macro_std = MaF1s_np.std() * 100
    s = f'MiF1={micro_mean:.2f}+-{micro_std:.2f}, MaF1={macro_mean:.2f}+-{macro_std:.2f}'
    logging.info(f'Evaluation stats: {s}')


def evaluate_graph_classification(
    Z: Tensor,
    y: Tensor,
    num_folds: int = 10,
    repeat: int = 1,
    seed: int = 0,
    C_grid: Optional[list[float]] = None,
) -> dict:
    """Standard TU graph-classification probe: 10-fold linear SVM.

    Standardizes features, sweeps SVM C per train split via inner CV, and reports
    mean/std accuracy across folds (optionally averaged over `repeat` seeds).
    Return shape mirrors `evaluate_linear_probe`.
    """
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X = Z.numpy()
    Y = y.numpy()
    C_grid = C_grid or [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]

    accuracies: list[float] = []
    for r in tqdm(range(repeat)):
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed + r)
        for train_idx, test_idx in skf.split(X, Y):
            clf = GridSearchCV(
                make_pipeline(StandardScaler(), SVC(kernel='linear')),
                param_grid={'svc__C': C_grid},
                cv=5,
                refit=True,
            )
            clf.fit(X[train_idx], Y[train_idx])
            preds = clf.predict(X[test_idx])
            accuracies.append(accuracy_score(Y[test_idx], preds))

    accuracies_arr = np.asarray(accuracies)
    return {
        'probe_type': f'svm-linear-{num_folds}fold',
        'mean': float(accuracies_arr.mean()),
        'std': float(accuracies_arr.std()),
        'accuracies': accuracies_arr.tolist(),
    }


def evaluate(
    embeddings: Tensor,
    full_data: Data,
    repeat: int,
    data_random_seed: int,
    dataset: str,
    full_eval: bool = False,
) -> dict:
    """Evaluate function for LeGraph node classification."""
    results = evaluate_linear_probe(
        embeddings,
        full_data,
        repeat=repeat,
        data_random_seed=data_random_seed,
    )

    if full_eval:
        logging.info(f'Full Evaluation.')
        has_masks = (
            hasattr(full_data, 'train_mask')
            and hasattr(full_data, 'val_mask')
            and hasattr(full_data, 'test_mask')
            and full_data.train_mask is not None
        )
        if has_masks:
            masks = (full_data.train_mask, full_data.val_mask, full_data.test_mask)
        else:
            masks = None
        node_classification(Z=embeddings, Y=full_data.y, dataset=dataset, masks=masks)

    return results
