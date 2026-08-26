"""Linear-probe evaluation of a frozen LeGTJEPA encoder on MM-Graph NC targets.

Cross-dataset transfer: the model is pretrained on the LP graphs
(sports-copurchase, cloth-copurchase, books-lp) and evaluated here on the two
NC graphs (ele-fashion, books-nc). The encoder is never trained on the targets.

Protocol (LeVLJEPA Sec. 5.2, transplanted to the graph embedding): freeze the
encoder, embed every node's ego-subgraph once to Z in R^{N x d}, then fit a
single linear layer Z[train] -> label and report test accuracy. Only the probe
trains; the encoder is frozen, so accuracy is a readout of the pretrained
features, not of target adaptation. The dataset's own split.pt gives the
train/val/test node masks; nodes with the missing-label sentinel are dropped
from all three. Repeated over data_args.eval_seeds (probe re-init only, since
the embedding is deterministic); mean accuracy +/- std.

No class-name text is used, so this needs no label vocabulary -- the reason we
use it in place of text-prompt zero-shot NC for MM-Graph.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch
from torch import Tensor
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader

from tgfm.dataset.evaluation.mm_load import load_mm_data, parse_mm_target_data
from tgfm.models.legtjepa import LeGTJEPA
from tgfm.utils.args import LeGTJEPAArguments, parse_args
from tgfm.utils.logger import setup_logging
from tgfm.utils.path import get_root_dir

torch.backends.mha.set_fastpath_enabled(False)

# labels-w-missing.pt marks absent labels with a sentinel; MM-Graph uses -1.
MISSING_LABEL = -1

parser = argparse.ArgumentParser(
    description='Linear-probe NC evaluation of frozen LeGTJEPA on MM-Graph.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file', type=str, required=True, help='Path to configuration file.'
)


@torch.no_grad()
def embed_all_nodes(
    model: torch.nn.Module,
    graphs: List,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    """Frozen graph embedding for every node's ego-subgraph. Z in R^{N x d}.

    Restores the caller's training mode on exit -- without this, an in-loop
    probe silently leaves BatchNorm in inference mode and dropout off for the
    remainder of training.
    """
    was_training = model.training
    model.eval()
    try:
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
        out = []
        for batch in loader:
            batch = batch.to(device)
            out.append(model.encode_graph(batch).cpu())
        return torch.cat(out, dim=0)
    finally:
        if was_training:
            model.train()


def fit_linear_probe(
    z_train: Tensor,
    y_train: Tensor,
    z_test: Tensor,
    y_test: Tensor,
    num_classes: int,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
) -> float:
    """Standardized-feature linear probe (LeVLJEPA App. C). Returns test acc."""
    mu, sd = z_train.mean(0, keepdim=True), z_train.std(0, keepdim=True).clamp_min(1e-8)
    z_train = ((z_train - mu) / sd).to(device)
    z_test = ((z_test - mu) / sd).to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)

    probe = torch.nn.Linear(z_train.size(1), num_classes, bias=True).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    probe.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(probe(z_train), y_train).backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        pred = probe(z_test).argmax(dim=1)
        return float((pred == y_test).float().mean())


def evaluate_dataset(
    model: torch.nn.Module,
    name: str,
    model_args: LeGTJEPAArguments,
    seeds: List[int],
    eval_batch_size: int,
    device: torch.device,
    feat_name: str,
) -> Tuple[float, float]:
    was_training = model.training
    data, _, _ = load_mm_data(name, feat_name=feat_name)
    graphs = parse_mm_target_data(name, data)

    # Encoder is frozen and deterministic: embed once, reuse across seeds.
    z = embed_all_nodes(model, graphs, device, eval_batch_size)
    y = data.y

    valid = y != MISSING_LABEL
    train_m = data.train_mask & valid
    test_m = data.test_mask & valid
    num_classes = int(y[valid].max().item()) + 1

    z_train, y_train = z[train_m], y[train_m]
    z_test, y_test = z[test_m], y[test_m]

    accs = []
    for seed in seeds:
        seed_everything(seed)  # varies probe init only
        accs.append(
            fit_linear_probe(z_train, y_train, z_test, y_test, num_classes, device)
        )
    acc_t = torch.tensor(accs)
    if was_training:
        model.train()
    return float(acc_t.mean()), float(acc_t.std())


def main() -> None:
    root = get_root_dir()
    args = parser.parse_args()
    meta_args, experiment_args = parse_args(root / args.config_file)
    for experiment, experiment_arg in experiment_args.exp_args.items():
        model_args = experiment_arg.model_args
        data_args = experiment_arg.data_args
        assert isinstance(model_args, LeGTJEPAArguments)
        setup_logging(meta_args.log_file_path)

        device = torch.device(
            f'cuda:{model_args.device}' if torch.cuda.is_available() else 'cpu'
        )
        model = LeGTJEPA(model_args).to(device)
        ckpt_path = str(
            Path(str(meta_args.root_dir)) / 'weights' / experiment / 'legtjepa.pt'
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        logging.info('Loaded %s (epoch %d)', ckpt_path, ckpt['epoch'])

        feat_name = getattr(model_args, 'mm_feat_name', 't5dino')
        results = {}
        for name in data_args.target_data.split('+'):
            mean, std = evaluate_dataset(
                model,
                name,
                model_args,
                data_args.eval_seeds,
                data_args.eval_batch_size,
                device,
                feat_name,
            )
            results[name] = (mean, std)
            logging.info(
                '%s linear-probe acc: %.2f +/- %.2f (feat=%s, seeds=%s)',
                name,
                100 * mean,
                100 * std,
                feat_name,
                data_args.eval_seeds,
            )
        macro = sum(m for m, _ in results.values()) / len(results)
        logging.info('Macro-averaged linear-probe acc: %.2f', 100 * macro)


if __name__ == '__main__':
    main()
