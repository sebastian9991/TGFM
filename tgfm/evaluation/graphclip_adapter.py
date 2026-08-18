"""Adapter: evaluate the released GraphCLIP checkpoint under our eval protocol.

GraphCLIP ships a pretrained checkpoint (README Sec. 3), so no retraining is
needed — only an interface shim. Our evaluation scripts call

    model.encode_graph(batch)                       -> (B, d)
    model.encode_text(input_ids, attention_mask)    -> (K, d)
    model.graph_predictor / model.text_predictor    -> R^d -> R^d

whereas GraphCLIP exposes ``encode_graph`` returning a
``(graph_embs, center_embs)`` tuple and ``encode_text(input_ids,
token_type_ids, attention_mask)``, and has no predictors. This wrapper
reconciles the two.

Scoring direction. GraphCLIP has no cross-modal predictors, so the only
admissible direction is ``direct``: cos(z_g, z_t). That is also the correct
one — InfoNCE is a function of the cross-modal Gram Z^g Z^t^T, which is
exactly what this score reads. The identity predictors below make
``zeroshot_direction`` a no-op rather than an error.

Data pipeline. Both models consume ego-subgraphs from the same
``parse_target_data``, so the perturbed inputs handed to each encoder are
literally the same objects — the property that makes the ImageNet-9-style
comparison in LeVLJEPA Sec. 5.3 controlled.

Validation gate. Their ``eval.py`` loads with ``strict=False``, which silently
tolerates missing keys; a partially-loaded graph tower would leave randomly
initialized weights and make any robustness number meaningless. This wrapper
logs missing/unexpected keys and refuses to load if graph-tower weights are
absent. Reproduce their Table 2 clean accuracies (Cora 67.31, CiteSeer 63.13,
WikiCS 70.19, Instagram 64.05, Photo 53.40, Computers 62.04, History 53.88)
before running a single attack.

Setup:
    git clone https://github.com/ZhuYun97/GraphCLIP third_party/GraphCLIP
    # download the released checkpoint into third_party/GraphCLIP/checkpoints/
    export PYTHONPATH=$PYTHONPATH:third_party/GraphCLIP
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class _Identity(nn.Module):
    """Stand-in for the cross-modal predictors GraphCLIP does not have."""

    def forward(self, z: Tensor) -> Tensor:
        return z


class GraphCLIPAdapter(nn.Module):
    """Wraps ``models.GraphCLIP`` in the interface our eval scripts expect."""

    def __init__(
        self,
        checkpoint_path: str,
        graph_input_dim: int = 384,
        graph_hid_dim: int = 1024,
        graph_num_layer: int = 12,
        text_model: str = 'tiny',
        attn_dropout: float = 0.0,
        strict: bool = False,
    ) -> None:
        super().__init__()
        from models import GraphCLIP  # requires the GraphCLIP repo on PYTHONPATH

        self.backbone = GraphCLIP(
            graph_input_dim,
            graph_hid_dim,
            graph_num_layer,
            {'dropout': attn_dropout},
            text_model=text_model,
        )
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        incompatible = self.backbone.load_state_dict(state, strict=strict)

        missing_graph = [k for k in incompatible.missing_keys if k.startswith('graph_model')]
        if missing_graph:
            raise RuntimeError(
                f'{len(missing_graph)} graph-tower keys missing from {checkpoint_path} '
                f'(first: {missing_graph[:3]}). The graph encoder would be partly '
                f'randomly initialized; refusing to proceed.'
            )
        logging.info(
            'GraphCLIP loaded from %s (%d missing, %d unexpected keys; text-tower '
            'keys may legitimately be missing since it is frozen and re-downloaded)',
            checkpoint_path,
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )

        # Our scripts branch on these; identity keeps `direct` the only real path.
        self.graph_predictor = _Identity()
        self.text_predictor = _Identity()

    def encode_graph(self, batch) -> Tensor:
        graph_embs, _center_embs = self.backbone.encode_graph(batch)
        return graph_embs  # their eval.py scores the pooled embedding

    def encode_text(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        return self.backbone.encode_text(input_ids, token_type_ids, attention_mask)

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)


def load_graphclip(
    checkpoint_path: str,
    device: torch.device,
    text_model: str = 'tiny',
) -> Tuple[GraphCLIPAdapter, str]:
    """Returns the adapter in eval mode and the tokenizer id to use with it."""
    model = GraphCLIPAdapter(checkpoint_path, text_model=text_model).to(device)
    model.eval()
    # Their eval.py hardcodes the MiniLM tokenizer regardless of --lm_type;
    # keep that behaviour so our numbers match theirs.
    return model, 'sentence-transformers/all-MiniLM-L6-v2'


def sanity_check(checkpoint_path: str) -> None:
    """Print the checkpoint's key inventory without instantiating the model."""
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    prefixes = {}
    for key in state:
        prefixes[key.split('.')[0]] = prefixes.get(key.split('.')[0], 0) + 1
    for prefix, count in sorted(prefixes.items()):
        print(f'{prefix:20s} {count:5d} tensors')


if __name__ == '__main__':
    import sys
    sanity_check(sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/pretrained_graphclip.pt')
