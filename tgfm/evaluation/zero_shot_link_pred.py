"""Zero-shot link prediction on target TAGs (GraphCLIP Table 3 protocol).

GraphCLIP reports mean AUC +/- std over five seeds, with 50% of edges held
out as test samples, using the pretrained model with no additional training.
Their released code does not include a link-prediction script; this
reproduces the protocol described in Sec. 4.3.2 of the paper.

Scoring. ``parse_target_data`` returns one ego-subgraph per node, ordered by
node id, so embedding the full list yields Z in R^{N x d} with row u the
embedding of node u's ego-subgraph. A candidate link (u, v) is scored by
cosine similarity cos(z_u, z_v). Positives are true edges; an equal number
of negatives is drawn uniformly from non-edges. AUC is computed over the
union.

Note both operands come from the *same* encoder, so unlike zero-shot node
classification this score needs no predictor to be well defined: the
relative-rotation gauge freedom of the LeGTJEPA objective cancels when both
sides are mapped identically. ``direct`` is therefore the principled default;
``graph_pred`` (scoring in text space) is offered only as a check.

Leakage caveat. Node u's ego-subgraph contains the edge (u, v) whenever v is
a neighbor, so a test edge is visible in the encoder input that produces its
own score. This is inherent to the GraphCLIP protocol and applies equally to
their reported numbers, so the comparison is controlled; ``--mask-test-edges``
re-parses the target graphs with test edges deleted for an honest-but-not-
comparable variant, reported alongside rather than instead.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.nn.functional import normalize
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.utils import coalesce, negative_sampling, to_undirected

from tgfm.dataset.evaluation.load import load_data
from tgfm.models.legtjepa import LeGTJEPA
from tgfm.utils.args import LeGTJEPAArguments, parse_args
from tgfm.utils.logger import setup_logging
from tgfm.utils.path import get_root_dir
from tgfm.utils.process import parse_target_data

# Optimized MHA fast path hits an illegal memory access on these batches.
torch.backends.mha.set_fastpath_enabled(False)

parser = argparse.ArgumentParser(
    description='Zero-shot link prediction evaluation of LeGTJEPA.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file', type=str, required=True, help='Path to configuration file.'
)
parser.add_argument(
    '--mask-test-edges',
    action='store_true',
    help='Delete test edges from the ego-subgraphs before encoding.',
)


@torch.no_grad()
def embed_all_nodes(
    model: torch.nn.Module,
    graphs: List,
    direction: str,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    """Encode every node's ego-subgraph. Returns Z in R^{N x d}, row-ordered."""
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    embeddings = []
    for batch in loader:
        batch = batch.to(device)
        z_g = model.encode_graph(batch)
        if direction == 'graph_pred':
            z_g = model.graph_predictor(z_g)
        embeddings.append(z_g.cpu())
    return torch.cat(embeddings, dim=0)


def undirected_edge_list(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Deduplicated (u, v) pairs with u < v, so each link is scored once."""
    edge_index = coalesce(to_undirected(edge_index), num_nodes=num_nodes)
    mask = edge_index[0] < edge_index[1]
    return edge_index[:, mask]


def link_auc(
    z: Tensor,
    pos_edges: Tensor,
    neg_edges: Tensor,
) -> float:
    z = normalize(z, dim=-1)  # cosine similarity; see LeVLJEPA Sec. 2.1
    pos_scores = (z[pos_edges[0]] * z[pos_edges[1]]).sum(-1)
    neg_scores = (z[neg_edges[0]] * z[neg_edges[1]]).sum(-1)
    scores = torch.cat([pos_scores, neg_scores]).float().numpy()
    labels = torch.cat(
        [
            torch.ones(pos_scores.numel()),
            torch.zeros(neg_scores.numel()),
        ]
    ).numpy()
    return float(roc_auc_score(labels, scores))


def evaluate_dataset(
    model: torch.nn.Module,
    name: str,
    model_args: LeGTJEPAArguments,
    seeds: List[int],
    test_ratio: float,
    eval_batch_size: int,
    device: torch.device,
    mask_test_edges: bool,
) -> Tuple[float, float]:
    data, _, _, _ = load_data(name, seed=0)
    num_nodes = data.num_nodes
    all_edges = undirected_edge_list(data.edge_index, num_nodes)
    num_test = int(test_ratio * all_edges.size(1))

    graphs = None
    z = None
    if not mask_test_edges:
        # Encoder input is seed-independent, so embed once and vary only the
        # split and the sampled negatives across seeds.
        graphs = parse_target_data(name, data)
        z = embed_all_nodes(
            model, graphs, model_args.zeroshot_direction, device, eval_batch_size
        )

    aucs = []
    for seed in seeds:
        seed_everything(seed)
        perm = torch.randperm(all_edges.size(1))
        pos_edges = all_edges[:, perm[:num_test]]
        neg_edges = negative_sampling(
            edge_index=to_undirected(data.edge_index),
            num_nodes=num_nodes,
            num_neg_samples=pos_edges.size(1),
            method='sparse',
        )

        if mask_test_edges:
            keep = perm[num_test:]
            kept = all_edges[:, keep]
            masked = data.clone()
            masked.edge_index = to_undirected(kept)
            z = embed_all_nodes(
                model,
                parse_target_data(name, masked),
                model_args.zeroshot_direction,
                device,
                eval_batch_size,
            )

        assert z is not None
        aucs.append(link_auc(z, pos_edges, neg_edges))

    auc_t = torch.tensor(aucs)
    return float(auc_t.mean()), float(auc_t.std())


def main() -> None:
    root = get_root_dir()
    args = parser.parse_args()
    config_file_path = root / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)

    for experiment, experiment_arg in experiment_args.exp_args.items():
        model_args = experiment_arg.model_args
        data_args = experiment_arg.data_args
        assert isinstance(model_args, LeGTJEPAArguments)
        setup_logging(meta_args.log_file_path)

        device = torch.device(
            f'cuda:{model_args.device}' if torch.cuda.is_available() else 'cpu'
        )
        model = LeGTJEPA(model_args).to(device)
        ckpt_path = (
            Path(str(meta_args.root_dir)) / 'weights' / experiment / 'legtjepa.pt'
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        logging.info('Loaded %s (epoch %d)', ckpt_path, ckpt['epoch'])

        results: Dict[str, Tuple[float, float]] = {}
        for name in data_args.target_data.split('+'):
            mean, std = evaluate_dataset(
                model=model,
                name=name,
                model_args=model_args,
                seeds=data_args.eval_seeds,
                test_ratio=0.5,  # GraphCLIP Sec. 4.3.2: 50% test edges
                eval_batch_size=data_args.eval_batch_size,
                device=device,
                mask_test_edges=args.mask_test_edges,
            )
            results[name] = (mean, std)
            logging.info(
                '%s link-pred AUC: %.2f +/- %.2f (direction=%s, masked=%s, seeds=%s)',
                name,
                100 * mean,
                100 * std,
                model_args.zeroshot_direction,
                args.mask_test_edges,
                data_args.eval_seeds,
            )

        macro = sum(m for m, _ in results.values()) / len(results)
        logging.info('Macro-averaged link-pred AUC: %.2f', 100 * macro)


if __name__ == '__main__':
    main()
