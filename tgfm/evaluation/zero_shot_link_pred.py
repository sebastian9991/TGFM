"""Zero-shot link prediction on target TAGs (GraphCLIP Table 3 protocol).

GraphCLIP reports mean AUC +/- std over five seeds, 50% of edges held out as
test, using the pretrained model with no additional training. Their released
code has no link-prediction script; this reproduces Sec. 4.3.2.

Two metrics, selected with ``--evaluation-method``:

  AUC  Rank a true edge against one uniformly sampled non-edge; probability the
       true edge scores higher. Answers "is a real edge above a random one?".
       This is GraphCLIP's reported metric.

  MRR  Mean Reciprocal Rank (filtered). For each positive (u, v), score v
       against ``--num-neg`` corrupted targets (u, v') where v' is not a real
       neighbour of u, rank the true target among the 1 + num_neg candidates,
       and average 1 / rank. Answers "how near the top is the real edge?" -- a
       retrieval-style question, strictly harder than AUC. Hits@10 is reported
       alongside. MRR is not comparable across different --num-neg, so hold it
       fixed across every model you compare.

Scoring. ``parse_target_data`` returns one ego-subgraph per node, ordered by
node id, so embedding the list yields Z in R^{N x d} with row u the embedding
of node u's ego-subgraph. A candidate link (u, v) scores as cos(z_u, z_v).
Both operands come from the same encoder, so this needs no predictor: the
relative-rotation gauge freedom of the LeGTJEPA objective cancels when both
sides are mapped identically. ``direct`` is the principled default;
``graph_pred`` (scoring in text space) is offered only as a check.

Leakage caveat. Node u's ego-subgraph contains edge (u, v) whenever v is a
neighbour, so a test edge is visible in the input that produces its own score.
Inherent to the GraphCLIP protocol and applied equally to their numbers, so
the comparison is controlled; ``--mask-test-edges`` deletes test edges from
the ego-subgraphs before encoding for an honest-but-not-comparable variant.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

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

torch.backends.mha.set_fastpath_enabled(False)

parser = argparse.ArgumentParser(
    description='Zero-shot link prediction evaluation of LeGTJEPA.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file', type=str, required=True, help='Path to configuration file.'
)
parser.add_argument(
    '--evaluation-method',
    choices=['AUC', 'MRR'],
    default='AUC',
    help='AUC (GraphCLIP Table 3) or filtered MRR + Hits@10.',
)
parser.add_argument(
    '--num-neg',
    type=int,
    default=100,
    help='Negatives per positive for MRR. Hold fixed across compared models.',
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
        z_g = model.graph_representations(batch)['backbone']
        if direction == 'graph_pred':
            z_g = model.graph_predictor(z_g)
        embeddings.append(z_g.cpu())
    return torch.cat(embeddings, dim=0)


def undirected_edge_list(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Deduplicated (u, v) pairs with u < v, so each link is scored once."""
    edge_index = coalesce(to_undirected(edge_index), num_nodes=num_nodes)
    mask = edge_index[0] < edge_index[1]
    return edge_index[:, mask]


def adjacency_set(edge_index: Tensor) -> Set[Tuple[int, int]]:
    """Directed pair set for O(1) filtered-negative rejection."""
    return {(int(a), int(b)) for a, b in edge_index.t().tolist()}


def link_auc(z: Tensor, pos_edges: Tensor, neg_edges: Tensor) -> Tuple[float, float]:
    """ROC-AUC over matched positive/negative edge sets. Second value is 0.0
    (no Hits@10 for AUC) so the return shape matches link_mrr."""
    z = normalize(z, dim=-1)  # cosine; embeddings not L2-normed in training
    pos_scores = (z[pos_edges[0]] * z[pos_edges[1]]).sum(-1)
    neg_scores = (z[neg_edges[0]] * z[neg_edges[1]]).sum(-1)
    scores = torch.cat([pos_scores, neg_scores]).float().numpy()
    labels = torch.cat(
        [torch.ones(pos_scores.numel()), torch.zeros(neg_scores.numel())]
    ).numpy()
    return float(roc_auc_score(labels, scores)), 0.0


def link_mrr(
    z: Tensor,
    pos_edges: Tensor,
    num_neg_per_pos: int,
    num_nodes: int,
    adj_set: Set[Tuple[int, int]],
    device: torch.device,
) -> Tuple[float, float]:
    """Filtered MRR and Hits@10.

    For each positive (u, v): score v against num_neg_per_pos corrupted targets
    (u, v') with v' not a real neighbour of u (filtered), rank the true target
    among 1 + num_neg candidates; reciprocal rank is 1 / rank.
    """
    z = normalize(z, dim=-1).to(device)
    reciprocal_ranks: List[float] = []
    hits10: List[float] = []
    for i in range(pos_edges.size(1)):
        u, v = int(pos_edges[0, i]), int(pos_edges[1, i])
        negs: List[int] = []
        while len(negs) < num_neg_per_pos:
            cand = int(torch.randint(0, num_nodes, (1,)))
            if cand != u and (u, cand) not in adj_set and (cand, u) not in adj_set:
                negs.append(cand)
        candidates = torch.tensor([v] + negs, device=device)
        scores = z[u] @ z[candidates].T  # (1 + num_neg,)
        rank = 1 + int((scores[1:] > scores[0]).sum())
        reciprocal_ranks.append(1.0 / rank)
        hits10.append(1.0 if rank <= 10 else 0.0)
    return (
        float(sum(reciprocal_ranks) / len(reciprocal_ranks)),
        float(sum(hits10) / len(hits10)),
    )


def evaluate_dataset(
    model: torch.nn.Module,
    name: str,
    model_args: LeGTJEPAArguments,
    seeds: List[int],
    test_ratio: float,
    eval_batch_size: int,
    device: torch.device,
    mask_test_edges: bool,
    evaluation_method: str,
    num_neg: int,
) -> Tuple[float, float, float]:
    """Returns (primary_mean, primary_std, secondary_mean). Primary is AUC or
    MRR; secondary is Hits@10 for MRR, else 0.0."""
    data, _, _, _ = load_data(name, seed=0)
    num_nodes = data.num_nodes
    all_edges = undirected_edge_list(data.edge_index, num_nodes)
    num_test = int(test_ratio * all_edges.size(1))
    adj_set = adjacency_set(to_undirected(data.edge_index))

    z = None
    if not mask_test_edges:
        # Encoder input is seed-independent: embed once, vary only the split
        # and sampled negatives across seeds.
        z = embed_all_nodes(
            model, parse_target_data(name, data),
            model_args.zeroshot_direction, device, eval_batch_size,
        )

    primary: List[float] = []
    secondary: List[float] = []
    for seed in seeds:
        seed_everything(seed)
        perm = torch.randperm(all_edges.size(1))
        pos_edges = all_edges[:, perm[:num_test]]

        if mask_test_edges:
            kept = all_edges[:, perm[num_test:]]
            masked = data.clone()
            masked.edge_index = to_undirected(kept)
            z = embed_all_nodes(
                model, parse_target_data(name, masked),
                model_args.zeroshot_direction, device, eval_batch_size,
            )
        assert z is not None

        if evaluation_method == 'AUC':
            neg_edges = negative_sampling(
                edge_index=to_undirected(data.edge_index),
                num_nodes=num_nodes,
                num_neg_samples=pos_edges.size(1),
                method='sparse',
            )
            p, s = link_auc(z, pos_edges, neg_edges)
        else:  # MRR
            p, s = link_mrr(z, pos_edges, num_neg, num_nodes, adj_set, device)
        primary.append(p)
        secondary.append(s)

    prim = torch.tensor(primary)
    return float(prim.mean()), float(prim.std()), float(torch.tensor(secondary).mean())


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
        ckpt_path = ( 
            Path(str(meta_args.root_dir)) / 'weights' / experiment / 'legtjepa.pt'
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        logging.info('Loaded %s (epoch %d)', ckpt_path, ckpt['epoch'])

        method = args.evaluation_method
        results: Dict[str, Tuple[float, float, float]] = {}
        for name in data_args.target_data.split('+'):
            mean, std, sec = evaluate_dataset(
                model=model,
                name=name,
                model_args=model_args,
                seeds=data_args.eval_seeds,
                test_ratio=0.5,  # GraphCLIP Sec. 4.3.2: 50% test edges
                eval_batch_size=data_args.eval_batch_size,
                device=device,
                mask_test_edges=args.mask_test_edges,
                evaluation_method=method,
                num_neg=args.num_neg,
            )
            results[name] = (mean, std, sec)
            if method == 'MRR':
                logging.info(
                    '%s MRR: %.4f +/- %.4f | Hits@10: %.4f '
                    '(direction=%s, num_neg=%d, masked=%s, seeds=%s)',
                    name, mean, std, sec, model_args.zeroshot_direction,
                    args.num_neg, args.mask_test_edges, data_args.eval_seeds,
                )
            else:
                logging.info(
                    '%s AUC: %.2f +/- %.2f (direction=%s, masked=%s, seeds=%s)',
                    name, 100 * mean, 100 * std, model_args.zeroshot_direction,
                    args.mask_test_edges, data_args.eval_seeds,
                )

        macro = sum(m for m, _, _ in results.values()) / len(results)
        if method == 'MRR':
            macro_h10 = sum(s for _, _, s in results.values()) / len(results)
            logging.info(
                'Macro MRR: %.4f | Macro Hits@10: %.4f', macro, macro_h10
            )
        else:
            logging.info('Macro-averaged link-pred AUC: %.2f', 100 * macro)


if __name__ == '__main__':
    main()
