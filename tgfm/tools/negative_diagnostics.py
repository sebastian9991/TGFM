"""Measure whether InfoNCE's negatives are still teaching, and how many are wrong.

Two claims need separating, and each has its own number.

  (1) Most negatives stop contributing gradient early.
      For anchor i, InfoNCE's gradient with respect to the logit s_ij of a
      negative j is p_ij / tau with p_ij = softmax_j(s_ij / tau). A negative
      with p_ij ~ 0 moves nothing. The count of negatives actually carrying
      gradient is the perplexity of that distribution,

          N_eff(i) = exp( H(p_i) ),    H in nats over the negatives only,

      which equals the number of negatives when the softmax is uniform and
      falls to ~1 when one negative dominates. Reported alongside the share
      of gradient mass held by the top-10 negatives.

  (2) Some negatives are semantically positives.
      Two nodes in a co-purchase catalogue can be the same product listed
      twice. InfoNCE repels them; the volume objective has no repulsion term
      and is indifferent. Counted per batch as pairs with feature cosine above
      tau, or, where labels exist, pairs sharing a label.

Modes
  --mode hook     print the patch for GraphCLIPLoss and exit. Logs (1) during
                  training, which is the measurement that carries the argument.
  --mode static   estimate (2) directly from precomputed node features by
                  drawing batches the way the sampler does. Needs no training.

    python tools/negative_diagnostics.py --mode hook
    python tools/negative_diagnostics.py --mode static \\
        --datasets sports-copurchase+cloth-copurchase+books-lp --batch 1800
"""

import argparse
import json
from typing import Dict

import torch
from torch import Tensor

parser = argparse.ArgumentParser(
    description='Effective-negative count and in-batch false-negative rate.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--mode', choices=['hook', 'static'], default='static')
parser.add_argument(
    '--datasets', type=str, default='sports-copurchase+cloth-copurchase+books-lp'
)
parser.add_argument('--batch', type=int, default=1800, help='Global batch size.')
parser.add_argument('--num-batches', type=int, default=200)
parser.add_argument('--dup-tau', type=float, default=0.95)
parser.add_argument('--mm-feat-name', type=str, default='t5dino')
parser.add_argument('--text-dim', type=int, default=768)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--out', type=str, default='')


@torch.no_grad()
def negative_stats(
    logits: Tensor,
    target: Tensor,
    partner: Tensor = None,
    dup_tau: float = 0.95,
    top: int = 10,
) -> Dict[str, float]:
    """Where InfoNCE's gradient goes, per anchor. logits is (B, B), pre-softmax.

    dL_i/ds_ij = p_ij for a negative and p_ii - 1 for the positive, with
    p = softmax_j(s_ij). So the three quantities that matter are:

      neg_grad_mass = 1 - p_ii      how much gradient reaches negatives AT ALL.
                                    Falls toward 0 as the positive comes to
                                    dominate. This, not N_eff, is what "the
                                    negatives stopped teaching" means.
      n_eff         = exp(H(p_neg)) how SPREAD that mass is over the negatives.
                                    High when negatives are indistinguishable,
                                    low when a few hard ones dominate -- so it
                                    is high in both the useful and the useless
                                    regime and must be read next to the mass.
      false_neg_grad_share          share of the negative mass sitting on
                                    negatives whose partner feature is within
                                    dup_tau of the anchor's. Gradient here is
                                    not merely wasted, it pushes apart pairs
                                    that are the same item. Needs `partner`,
                                    the (B, d) partner-modality features.
    """
    B = logits.shape[0]
    idx = torch.arange(B, device=logits.device)
    full = logits.softmax(dim=1)
    p_pos = full[idx, target]

    masked = logits.clone()
    masked[idx, target] = float('-inf')
    p = masked.softmax(dim=1)  # renormalised over negatives only
    n_eff = torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum(dim=1))
    topk = p.topk(min(top, B - 1), dim=1).values.sum(dim=1)
    rank = (masked > logits[idx, target].unsqueeze(1)).sum(dim=1)

    out = {
        'neg_grad_mass': float((1.0 - p_pos).mean()),
        'n_eff_mean': float(n_eff.mean()),
        'n_eff_frac': float(n_eff.mean()) / (B - 1),
        f'top{top}_grad_share': float(topk.mean()),
        'pos_rank_mean': float(rank.double().mean()),
        'acc_r1': float((rank == 0).double().mean()),
        'num_negatives': B - 1,
    }
    if partner is not None:
        u = torch.nn.functional.normalize(partner.float(), dim=-1)
        dup = (u @ u.t()) > dup_tau
        dup[idx, target] = False
        out['false_neg_per_anchor'] = float(dup.sum(dim=1).double().mean())
        out['false_neg_grad_share'] = float((p * dup).sum(dim=1).mean())
    return out


HOOK_PATCH = """
# --- tgfm/models/losses/graphclip_loss.py, inside GraphCLIPLoss.forward ---
# after `logits_g, logits_p = self._create_logits(z_g, z_p, scale)`:

        from tgfm.tools.negative_diagnostics import negative_stats

        target = torch.arange(logits_g.size(0), device=logits_g.device)
        # z_p is the frozen partner feature, so its cosines identify the pairs
        # that are the same item listed twice
        diag = negative_stats(logits_g, target, partner=z_p, dup_tau=0.95)
        # merged into the dict train_epoch already logs, so these reach the
        # existing log line and wandb with no further wiring
        out.update({k: torch.as_tensor(v) for k, v in diag.items()})

# What to read, over epoch 1:
#
#   neg_grad_mass          1 - p_positive. The run's logged infonce value
#                          already bounds this: CE = -log p_pos, so
#                          p_pos = exp(-infonce). At infonce = 1.13 that is
#                          p_pos = 0.32, i.e. 68% of gradient still reaching
#                          negatives -- they have NOT stopped teaching by the
#                          end of epoch 1, despite r1_g2p = 0.99. Ranking is
#                          solved long before the softmax saturates, so
#                          r1_g2p on its own does not support the claim.
#
#   false_neg_grad_share   the share of that mass spent pushing apart pairs
#                          that are the same item. This is the number the
#                          argument rests on: it makes the repulsion wrong
#                          rather than merely inefficient.
#
#   n_eff_frac             read only next to neg_grad_mass. It is high both
#                          when negatives are uniformly useless and when they
#                          are uniformly informative.
"""


def load_features(name: str, feat_name: str, text_dim: int) -> Tensor:
    """Per-node text features for an MM-Graph or TAG dataset."""
    try:
        from tgfm.dataset.evaluation.mm_load import load_mm_data

        data, _, _ = load_mm_data(name, feat_name=feat_name)
        x = data.x
        return x[:, :text_dim] if x.shape[1] > text_dim else x
    except Exception:  # noqa: BLE001 - fall through to the TAG loader
        from tgfm.dataset.evaluation.load import load_data

        data, _, _, _ = load_data(name)
        return data.x


@torch.no_grad()
def static_collisions(
    x: Tensor, B: int, n_batches: int, tau: float, device: torch.device, seed: int
) -> Dict[str, float]:
    """In-batch false-negative rate, from batches drawn as the sampler draws them.

    The sampler shuffles node ids, so a batch is a uniform sample without
    replacement. For each batch the full B x B cosine matrix is formed -- the
    same matrix InfoNCE builds -- and pairs above tau are counted off the
    diagonal. Reported per anchor so it can be read as "how many of my B-1
    negatives are actually positives".
    """
    N = x.shape[0]
    g = torch.Generator().manual_seed(seed)
    per_anchor, worst, any_dup = [], 0.0, 0
    for _ in range(n_batches):
        idx = torch.randperm(N, generator=g)[:B]
        z = torch.nn.functional.normalize(x[idx].to(device).float(), dim=-1)
        c = z @ z.t()
        c.fill_diagonal_(-1.0)
        hit = c > tau
        counts = hit.sum(dim=1).double()
        per_anchor.append(float(counts.mean()))
        worst = max(worst, float(counts.max()))
        any_dup += int((counts > 0).sum())
    mean = sum(per_anchor) / len(per_anchor)
    return {
        'false_neg_per_anchor': mean,
        'false_neg_per_batch': mean * B / 2,
        'max_per_anchor': worst,
        'frac_anchors_with_any': any_dup / (n_batches * B),
        'batch_size': B,
        'batches': n_batches,
    }


def main() -> None:
    args = parser.parse_args()
    if args.mode == 'hook':
        print(HOOK_PATCH)
        return

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    rows = []
    feats = []
    for name in args.datasets.split('+'):
        print(f'loading {name} ...')
        try:
            x = load_features(name, args.mm_feat_name, args.text_dim)
        except Exception as exc:  # noqa: BLE001
            print(f'  SKIP {name}: {type(exc).__name__}: {exc}')
            continue
        feats.append(x)
        stats = static_collisions(
            x, args.batch, args.num_batches, args.dup_tau, device, args.seed
        )
        rows.append({'dataset': name, 'nodes': x.shape[0], **stats})

    if len(feats) > 1:
        # Training batches are drawn from the ConcatDataset, so the pooled
        # number is the one the run actually sees. Reported separately because
        # a mixed pool dilutes within-domain duplicates.
        print('pooled ...')
        pooled = torch.cat(feats, dim=0)
        rows.append(
            {
                'dataset': 'POOLED',
                'nodes': pooled.shape[0],
                **static_collisions(
                    pooled,
                    args.batch,
                    args.num_batches,
                    args.dup_tau,
                    device,
                    args.seed,
                ),
            }
        )

    head = (
        f'{"dataset":<22}{"nodes":>9}{"fn/anchor":>11}{"fn/batch":>10}'
        f'{"max/anchor":>12}{"anchors hit":>13}'
    )
    print(f'\n=== in-batch false negatives (cos > {args.dup_tau}, B={args.batch}) ===')
    print(head)
    print('-' * len(head))
    for r in rows:
        print(
            f'{r["dataset"]:<22}{r["nodes"]:>9}{r["false_neg_per_anchor"]:>11.3f}'
            f'{r["false_neg_per_batch"]:>10.1f}{r["max_per_anchor"]:>12.0f}'
            f'{r["frac_anchors_with_any"]:>12.2%}'
        )
    print(
        '\nfn/batch is the number of semantically identical pairs InfoNCE pushes '
        f'apart at every step. Over one epoch that is multiplied by the step count.'
    )
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump({'args': vars(args), 'rows': rows}, fh, indent=2)
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
