"""Adversarial robustness of zero-shot node classification (LeVLJEPA Sec. 5.3 analog).

LeVLJEPA evaluates background robustness by training a probe on Original
ImageNet-9 features and evaluating, without retraining, on Mixed-Same and
Mixed-Rand recompositions. The graph analog replaces the background swap with
a bounded perturbation of the ego-subgraph and the linear probe with the
zero-shot readout, so nothing is retrained on the perturbed inputs. This is an
evasion attack in the sense of Zuegner et al. (KDD 2018): the encoder is
frozen and only the input is perturbed. Poisoning does not apply, since the
target graph is never trained on.

Attack surfaces (all bounded, all on the graph side):
    random_feat   uniform noise on node features, ||delta||_inf <= eps
    pgd_feat      L_inf PGD on node features against the zero-shot margin
    random_struct random edge flips, budget Delta
    rbcd_struct   randomized-block greedy edge flips against the margin, after
                  Geisler et al. (NeurIPS 2021); each round samples a block of
                  candidate flips and keeps the best one

Node features are SBERT sentence embeddings, not the binary bag-of-words
Zuegner et al. assume, so their feature co-occurrence test does not apply and
the feature budget is an L_inf ball instead. Their degree power-law test does
transfer unchanged and gates every structural flip (Eq. 6-10, tau = 0.004).

Comparability. Three knobs keep the two arms comparable, in the sense that
LeVLJEPA's Mixed-Rand images do not depend on which encoder is evaluated:
    --surrogate-model  generates perturbations against one fixed model
    --targets-file     pins the attacked node set across arms
    a shared --seed stream, so the perturbation RNG is identical
Run each arm with the same three and both encoders see the same perturbed
inputs. White-box (--surrogate-model same) measures worst-case robustness but
is not comparable across models; report both.

GraphCLIP is scored with direction 'direct' because it has no cross-modal
predictors -- and that is the correct readout for it, since InfoNCE is a
function of the cross-modal Gram. LeGTJEPA keeps its configured direction.
State the asymmetry in the paper: each method is scored by the quantity its
objective identifies.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn.functional import normalize
from torch_geometric import seed_everything
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tgfm.dataset.evaluation.load import load_data
from tgfm.evaluation.zero_shot_eval import encode_label_sentences
from tgfm.models.legtjepa import LeGTJEPA
from tgfm.utils.args import LeGTJEPAArguments, MetaArguments, parse_args
from tgfm.utils.logger import setup_logging
from tgfm.utils.path import get_root_dir
from tgfm.utils.process import parse_target_data

# Optimized MHA fast path hits an illegal memory access under eval + no_grad.
torch.backends.mha.set_fastpath_enabled(False)

ATTACKS = ('clean', 'random_feat', 'pgd_feat', 'random_struct', 'rbcd_struct')

parser = argparse.ArgumentParser(
    description='Adversarial robustness evaluation of LeGTJEPA vs GraphCLIP.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file', type=str, required=True, help='Path to configuration file.'
)
parser.add_argument(
    '--eval-model',
    choices=('legtjepa', 'graphclip'),
    default='legtjepa',
    help='Which model is evaluated on the perturbed inputs.',
)
parser.add_argument(
    '--surrogate-model',
    choices=('same', 'legtjepa', 'graphclip'),
    default='same',
    help="Which model perturbations are generated against; 'same' = white-box.",
)
parser.add_argument(
    '--graphclip-ckpt',
    type=str,
    default=None,
    help='Path to the released GraphCLIP checkpoint (.pt).',
)
parser.add_argument(
    '--targets-file',
    type=str,
    default=None,
    help='Directory caching target node indices; written if absent, read if '
    'present. Both arms must share one.',
)
parser.add_argument(
    '--num-targets', type=int, default=40, help='Target nodes per dataset.'
)
parser.add_argument(
    '--feat-eps', type=float, default=1.0e-2, help='L_inf feature budget.'
)
parser.add_argument(
    '--pgd-steps', type=int, default=20, help='PGD iterations for feature attacks.'
)
parser.add_argument(
    '--block-size', type=int, default=64, help='Candidate flips sampled per round.'
)
parser.add_argument(
    '--degree-tau',
    type=float,
    default=0.004,
    help='Power-law likelihood-ratio threshold (Zuegner et al. Eq. 10).',
)


def build_model(
    kind: str,
    model_args: LeGTJEPAArguments,
    meta_args: MetaArguments,
    experiment: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.nn.Module, PreTrainedTokenizerBase, str]:
    """Returns (model in eval mode, its tokenizer, its admissible direction)."""
    if kind == 'graphclip':
        if args.graphclip_ckpt is None:
            raise ValueError('--graphclip-ckpt is required for the graphclip arm.')
        from tgfm.evaluation.graphclip_adapter import load_graphclip

        model, tokenizer_id = load_graphclip(args.graphclip_ckpt, device)
        # No cross-modal predictors exist; `direct` is the only admissible score.
        return model, AutoTokenizer.from_pretrained(tokenizer_id), 'direct'

    model = LeGTJEPA(model_args).to(device)
    ckpt_path = (
        Path(str(meta_args.root_dir))
        / 'weights'
        / f'{experiment}--{meta_args.wandb_sweep_id}'
        / 'legtjepa.pt'
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    logging.info('Loaded %s (epoch %d)', ckpt_path, ckpt['epoch'])
    return (
        model,
        AutoTokenizer.from_pretrained(model_args.text_model_id),
        model_args.zeroshot_direction,
    )


def zero_shot_margin(
    model: torch.nn.Module, batch: Batch, z_labels: Tensor, direction: str
) -> Tensor:
    """Per-graph margin: true-class similarity minus best other-class similarity."""
    z_g = model.encode_graph(batch)
    if direction in ('graph_pred', 'text_pred+graph_pred'):
        z_g = model.graph_predictor(z_g)
    sims = normalize(z_g, dim=-1) @ z_labels.T
    true_sim = sims.gather(1, batch.y.view(-1, 1)).squeeze(1)
    other = sims.scatter(1, batch.y.view(-1, 1), float('-inf'))
    return true_sim - other.max(dim=1).values


def prepare_labels(model: torch.nn.Module, z_labels: Tensor, direction: str) -> Tensor:
    if direction in ('text_pred', 'text_pred+graph_pred'):
        z_labels = model.text_predictor(z_labels)
    return normalize(z_labels, dim=-1)


def label_embeddings(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    classes: List[str],
    c_descs: List[str],
    name: str,
    direction: str,
    device: torch.device,
    max_length: int,
) -> Tensor:
    return prepare_labels(
        model,
        encode_label_sentences(
            model, tokenizer, classes, c_descs, name, device, max_length
        ),
        direction,
    )


def powerlaw_alpha(degrees: Tensor, d_min: int = 2) -> Tuple[Tensor, Tensor]:
    """Scaling exponent and log-degree sum for the power-law test (Eq. 6)."""
    kept = degrees[degrees >= d_min].float()
    if kept.numel() == 0:
        return torch.tensor(float('nan')), torch.tensor(0.0)
    log_sum = torch.log(kept / (d_min - 0.5)).sum()
    alpha = 1.0 + kept.numel() / log_sum.clamp(min=1e-9)
    return alpha, torch.log(kept).sum()


def degree_ratio_test(deg_clean: Tensor, deg_pert: Tensor, d_min: int = 2) -> float:
    """Likelihood-ratio statistic Lambda (Eq. 9); small values are unnoticeable."""
    import math

    def ll(degrees: Tensor) -> Tensor:
        alpha, log_sum = powerlaw_alpha(degrees, d_min)
        n = (degrees >= d_min).sum().float()
        if n == 0 or torch.isnan(alpha):
            return torch.tensor(0.0)
        return (
            n * torch.log(alpha) + n * alpha * math.log(d_min) + (alpha + 1) * log_sum
        )

    combined = torch.cat([deg_clean, deg_pert])
    return float(-2.0 * ll(combined) + 2.0 * (ll(deg_clean) + ll(deg_pert)))


def random_feature_noise(batch: Batch, eps: float) -> Batch:
    perturbed = batch.clone()
    perturbed.x = batch.x + torch.empty_like(batch.x).uniform_(-eps, eps)
    return perturbed


def pgd_feature_attack(
    model: torch.nn.Module,
    batch: Batch,
    z_labels: Tensor,
    direction: str,
    eps: float,
    steps: int,
) -> Batch:
    """L_inf PGD on node features, minimizing the zero-shot margin."""
    x_clean = batch.x.detach()
    delta = torch.empty_like(x_clean).uniform_(-eps, eps).requires_grad_()
    step_size = 2.5 * eps / max(1, steps)

    for _ in range(steps):
        batch.x = x_clean + delta
        margin = zero_shot_margin(model, batch, z_labels, direction).sum()
        (grad,) = torch.autograd.grad(margin, delta)
        with torch.no_grad():
            delta -= step_size * grad.sign()  # descend the margin
            delta.clamp_(-eps, eps)
        delta.requires_grad_()

    perturbed = batch.clone()
    perturbed.x = (x_clean + delta).detach()
    batch.x = x_clean
    return perturbed


def sample_flips(
    graph, num_flips: int, generator: torch.Generator
) -> List[Tuple[int, int]]:
    n = graph.num_nodes
    if n < 2:
        return []
    u = torch.randint(0, n, (num_flips,), generator=generator)
    v = torch.randint(0, n, (num_flips,), generator=generator)
    return [(int(a), int(b)) for a, b in zip(u, v) if a != b]


def apply_flip(graph, u: int, v: int):
    """Toggle undirected edge (u, v). Returns a new Data object."""
    perturbed = graph.clone()
    ei = perturbed.edge_index
    mask = ~(((ei[0] == u) & (ei[1] == v)) | ((ei[0] == v) & (ei[1] == u)))
    if bool((~mask).any()):  # edge present -> delete
        perturbed.edge_index = ei[:, mask]
    else:  # edge absent -> insert both directions
        new = torch.tensor([[u, v], [v, u]], device=ei.device, dtype=ei.dtype)
        perturbed.edge_index = torch.cat([ei, new], dim=1)
    return perturbed


def degrees_of(graph) -> Tensor:
    return torch.bincount(graph.edge_index[0], minlength=graph.num_nodes)


@torch.no_grad()
def structure_attack(
    model: torch.nn.Module,
    graph,
    z_labels: Tensor,
    direction: str,
    budget: int,
    block_size: int,
    degree_tau: float,
    device: torch.device,
    generator: torch.Generator,
    greedy: bool,
):
    """Edge-flip attack under a degree-preserving unnoticeability constraint.

    ``greedy=False`` accepts a random admissible flip each round (the Rnd
    baseline of Zuegner et al.); ``greedy=True`` scores a random block of
    candidates and keeps the margin-minimizing one, the block-coordinate
    scheme of Geisler et al. applied per ego-subgraph.
    """
    current = graph
    deg_clean = degrees_of(graph)

    for _ in range(budget):
        candidates = sample_flips(current, block_size, generator)
        admissible = []
        for u, v in candidates:
            proposal = apply_flip(current, u, v)
            if degree_ratio_test(deg_clean, degrees_of(proposal)) < degree_tau:
                admissible.append(proposal)
        if not admissible:
            break
        if not greedy:
            current = admissible[0]
            continue
        scored = Batch.from_data_list(admissible).to(device)
        margins = zero_shot_margin(model, scored, z_labels, direction)
        current = admissible[int(margins.argmin())]

    return current


def select_targets(
    margins: Tensor, num_targets: int, generator: torch.Generator
) -> Tensor:
    """Zuegner et al. Sec. 6: highest-margin, lowest-margin, and random correct nodes."""
    correct = (margins > 0).nonzero().flatten()
    if correct.numel() <= num_targets:
        return correct
    order = correct[margins[correct].argsort(descending=True)]
    n_extreme = num_targets // 4
    high, low = order[:n_extreme], order[-n_extreme:]
    remaining = order[n_extreme:-n_extreme]
    perm = torch.randperm(remaining.numel(), generator=generator)
    rest = remaining[perm[: num_targets - 2 * n_extreme]]
    return torch.cat([high, low, rest])


def resolve_targets(
    clean_margins: Tensor,
    name: str,
    seed: int,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> Tensor:
    """Shared target set across arms: the perturbed inputs must not depend on
    which encoder is being evaluated (LeVLJEPA Sec. 5.3 uses identical images
    for every backbone).
    """
    cache = (
        Path(args.targets_file) / f'{name}_seed{seed}.pt' if args.targets_file else None
    )
    if cache is not None and cache.exists():
        targets = torch.load(cache)
        logging.info(
            '%s: reusing %d cached targets from %s', name, targets.numel(), cache
        )
        return targets

    targets = select_targets(clean_margins, args.num_targets, generator)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(targets, cache)
        logging.info('%s: wrote %d targets to %s', name, targets.numel(), cache)
    return targets


def run_dataset(
    model: torch.nn.Module,
    tok_eval: PreTrainedTokenizerBase,
    dir_eval: str,
    surrogate: torch.nn.Module,
    tok_att: PreTrainedTokenizerBase,
    dir_att: str,
    model_args: LeGTJEPAArguments,
    name: str,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> Dict[str, Tuple[float, float]]:
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)

    data, _, classes, c_descs = load_data(name, seed=0)
    graphs = parse_target_data(name, data)

    with torch.no_grad():
        z_labels_eval = label_embeddings(
            model,
            tok_eval,
            classes,
            c_descs,
            name,
            dir_eval,
            device,
            model_args.max_text_length,
        )
        z_labels_att = (
            z_labels_eval
            if surrogate is model
            else label_embeddings(
                surrogate,
                tok_att,
                classes,
                c_descs,
                name,
                dir_att,
                device,
                model_args.max_text_length,
            )
        )

        clean_margins = []
        for batch in DataLoader(graphs, batch_size=256, shuffle=False):
            batch = batch.to(device)
            clean_margins.append(
                zero_shot_margin(model, batch, z_labels_eval, dir_eval).cpu()
            )
        clean_margins = torch.cat(clean_margins)

    targets = resolve_targets(clean_margins, name, seed, args, generator)
    logging.info(
        '%s: %d targets; %d/%d correctly classified clean',
        name,
        targets.numel(),
        int((clean_margins > 0).sum()),
        clean_margins.numel(),
    )

    results: Dict[str, List[float]] = {a: [] for a in ATTACKS}
    for idx in targets.tolist():
        graph = graphs[idx]
        budget = int(degrees_of(graph)[graph.root_n_index]) + 2  # Nettack budget
        variants = {'clean': graph}

        single = Batch.from_data_list([graph]).to(device)
        with torch.no_grad():
            variants['random_feat'] = random_feature_noise(single, args.feat_eps)
        variants['pgd_feat'] = pgd_feature_attack(
            surrogate, single, z_labels_att, dir_att, args.feat_eps, args.pgd_steps
        )
        for key, greedy in (('random_struct', False), ('rbcd_struct', True)):
            variants[key] = structure_attack(
                surrogate,
                graph,
                z_labels_att,
                dir_att,
                budget,
                args.block_size,
                args.degree_tau,
                device,
                generator,
                greedy,
            )

        with torch.no_grad():
            for key, variant in variants.items():
                batch = (
                    variant
                    if isinstance(variant, Batch)
                    else Batch.from_data_list([variant]).to(device)
                )
                margin = zero_shot_margin(model, batch, z_labels_eval, dir_eval)
                results[key].append(float(margin.mean()))

    summary = {}
    for key, margins in results.items():
        m = torch.tensor(margins)
        summary[key] = (float((m > 0).float().mean()), float(m.mean()))
    return summary


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

        model, tok_eval, dir_eval = build_model(
            args.eval_model, model_args, meta_args, experiment, args, device
        )
        if args.surrogate_model in ('same', args.eval_model):
            surrogate, tok_att, dir_att = model, tok_eval, dir_eval
            mode = f'white-box ({args.eval_model})'
        else:
            surrogate, tok_att, dir_att = build_model(
                args.surrogate_model, model_args, meta_args, experiment, args, device
            )
            mode = f'transfer ({args.surrogate_model} -> {args.eval_model})'
        logging.info(
            'Robustness eval | eval=%s dir=%s | surrogate=%s dir=%s | %s',
            args.eval_model,
            dir_eval,
            args.surrogate_model,
            dir_att,
            mode,
        )

        for name in data_args.target_data.split('+'):
            per_seed: Dict[str, List[Tuple[float, float]]] = {a: [] for a in ATTACKS}
            for seed in data_args.eval_seeds:
                summary = run_dataset(
                    model,
                    tok_eval,
                    dir_eval,
                    surrogate,
                    tok_att,
                    dir_att,
                    model_args,
                    name,
                    args,
                    device,
                    seed,
                )
                for key, value in summary.items():
                    per_seed[key].append(value)

            clean_acc = torch.tensor([a for a, _ in per_seed['clean']]).mean()
            for key in ATTACKS:
                accs = torch.tensor([a for a, _ in per_seed[key]])
                margins = torch.tensor([m for _, m in per_seed[key]])
                logging.info(
                    '%s | %-9s | %-14s acc=%.2f +/- %.2f  margin=%+.4f  drop=%.2f',
                    name,
                    args.eval_model,
                    key,
                    100 * accs.mean(),
                    100 * accs.std(),
                    margins.mean(),
                    100 * (clean_acc - accs.mean()),
                )


if __name__ == '__main__':
    main()
