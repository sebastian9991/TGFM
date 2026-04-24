"""Implementation of UniGraph with memory optimizations.

Key changes vs. original:
  1. LM weights are SHARED between encoder and target (paper: δ' ← δ).
     We no longer maintain a second AutoModel — we just call the same LM
     under torch.no_grad() for the target path.
  2. EMA updates only the GNN (and fusion/projector), not the LM.
  3. Fixed the "graph_embedding_matrix" math: it now broadcasts each node's
     own GNN output to its own token positions (paper Eq. 3), instead of
     summing all node embeddings.
  4. MLM head is computed only on masked token positions — saves the
     [B, seq, vocab] logits tensor (biggest activation by far).
  5. Fusion uses two Linear layers with addition instead of concat+Linear.
     Equivalent math, lower peak activation memory.
  6. Optional gradient checkpointing on the LM (toggle via args).

@article{he2025unigraph, ...}
"""

from typing import Iterator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from tgfm.models.gat import GAT
from tgfm.utils.args import UnigraphArguments


class UniGraph(nn.Module):
    """UniGraph: Unified Cross-Domain Foundation Model for Text-Attributed Graphs."""

    def __init__(self, args: UnigraphArguments):
        super().__init__()
        self.args = args

        # Shared language model (used for BOTH online and target paths).
        self.lm_encoder = AutoModel.from_pretrained(args.lm_type)
        if getattr(args, 'gradient_checkpointing', False):
            self.lm_encoder.gradient_checkpointing_enable()

        hidden = args.hidden_size

        # Online GNN encoder
        self.gnn_encoder = GAT(
            in_dim=hidden,
            hidden_dim=hidden,
            out_dim=hidden,
            num_layers=args.num_layers,
            nhead=args.nhead,
            feat_drop=args.dropout,
        )

        # Fusion: two linears + add instead of concat + linear.
        # Equivalent to Linear(2d -> d) on cat([lm, gnn]) but with lower
        # peak activation memory (we never materialize the [B, seq, 2d] tensor).
        self.fuse_lm = nn.Linear(hidden, hidden)
        self.fuse_gnn = nn.Linear(hidden, hidden)
        self.fuse_norm = nn.LayerNorm(hidden)

        if args.lam > 0:
            self.projector = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )

            # Target networks — ONLY the GNN, fusion, and projector are
            # duplicated. The LM is shared (paper Sec. 4.2: δ' ← δ).
            self.target_gnn_encoder = GAT(
                in_dim=hidden,
                hidden_dim=hidden,
                out_dim=hidden,
                num_layers=args.num_layers,
                nhead=args.nhead,
                feat_drop=args.dropout,
            )
            self.target_fuse_lm = nn.Linear(hidden, hidden)
            self.target_fuse_gnn = nn.Linear(hidden, hidden)
            self.target_fuse_norm = nn.LayerNorm(hidden)
            self.target_projector = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )

            self._init_target_networks()

        # MLM head — applied only to masked token positions at forward time.
        self.mlm_head = nn.Linear(hidden, self.lm_encoder.config.vocab_size)

    def _init_target_networks(self) -> None:
        self.target_gnn_encoder.load_state_dict(self.gnn_encoder.state_dict())
        self.target_fuse_lm.load_state_dict(self.fuse_lm.state_dict())
        self.target_fuse_gnn.load_state_dict(self.fuse_gnn.state_dict())
        self.target_fuse_norm.load_state_dict(self.fuse_norm.state_dict())
        self.target_projector.load_state_dict(self.projector.state_dict())

        for p in self._target_parameters():
            p.requires_grad = False

    def _target_parameters(self) -> Iterator[torch.nn.Parameter]:
        yield from self.target_gnn_encoder.parameters()
        yield from self.target_fuse_lm.parameters()
        yield from self.target_fuse_gnn.parameters()
        yield from self.target_fuse_norm.parameters()
        yield from self.target_projector.parameters()

    def _online_parameters_for_ema(self) -> Iterator[torch.nn.Parameter]:
        yield from self.gnn_encoder.parameters()
        yield from self.fuse_lm.parameters()
        yield from self.fuse_gnn.parameters()
        yield from self.fuse_norm.parameters()
        yield from self.projector.parameters()

    @torch.no_grad()
    def _update_target_networks(self, tau: float = 0.99) -> None:
        """EMA update. Paper: ξ' ← τ·ξ' + (1 - τ)·ξ.

        We update in-place with mul_/add_ to avoid allocating new tensors
        every step (your original version did a fresh tensor per parameter).
        """
        for p_target, p_online in zip(
            self._target_parameters(), self._online_parameters_for_ema()
        ):
            p_target.data.mul_(tau).add_(p_online.data, alpha=1.0 - tau)

    def _broadcast_gnn_to_tokens(
        self, graph_embeddings: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """Broadcast per-node GNN output to every token position of that node.

        graph_embeddings: [B, d]  ->  [B, seq_len, d]  (view, no copy)
        """
        return graph_embeddings.unsqueeze(1).expand(-1, seq_len, -1)

    def _fuse(
        self,
        lm_hidden: torch.Tensor,  # [B, seq, d]
        graph_embeddings: torch.Tensor,  # [B, d]
        fuse_lm: nn.Linear,
        fuse_gnn: nn.Linear,
        fuse_norm: nn.LayerNorm,
    ) -> torch.Tensor:
        """Fuse LM hidden states with per-node GNN embeddings.

        Equivalent to Linear(2d -> d) applied to cat([lm, gnn_broadcast]),
        but avoids materializing the [B, seq, 2d] intermediate.
        """
        # fuse_gnn(graph_embeddings) is [B, d] -> [B, d]; broadcast to [B, seq, d]
        gnn_proj = fuse_gnn(graph_embeddings).unsqueeze(1)  # [B, 1, d]
        lm_proj = fuse_lm(lm_hidden)  # [B, seq, d]
        return F.relu(fuse_norm(lm_proj + gnn_proj))

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, seq]  (unmasked, for target & labels)
        masked_input_ids: torch.Tensor,  # [B, seq]
        attention_mask: torch.Tensor,  # [B, seq]
        token_type_ids: torch.Tensor,  # [B, seq]
        edge_index: torch.Tensor,  # [2, E]
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lm_out = self.lm_encoder(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        lm_hidden = lm_out.last_hidden_state  # [B, seq, d]
        node_features = lm_hidden[:, 0, :]  # [CLS] per node, [B, d]

        graph_embeddings = self.gnn_encoder(node_features, edge_index)  # [B, d]

        combined = self._fuse(
            lm_hidden,
            graph_embeddings,
            self.fuse_lm,
            self.fuse_gnn,
            self.fuse_norm,
        )  # [B, seq, d]

        # ---- MLM loss on masked positions only ----
        # This is the single biggest memory win: we never materialize
        # [B, seq, vocab] logits. At B=128, seq=512, vocab=30522, fp16:
        #   full:   128 * 512 * 30522 * 2 bytes ≈ 4.0 GB
        #   masked: ~15% of that ≈ 0.6 GB
        mask_positions = masked_input_ids == self._mask_token_id()  # [B, seq] bool
        # Guard against a batch with zero masked tokens (rare but possible).
        if mask_positions.any():
            masked_hidden = combined[mask_positions]  # [N_masked, d]
            masked_logits = self.mlm_head(masked_hidden)  # [N_masked, V]
            masked_labels = input_ids[mask_positions]  # [N_masked]
            mlm_loss = F.cross_entropy(masked_logits, masked_labels)
        else:
            mlm_loss = combined.sum() * 0.0  # keeps graph connected, zero loss

        latent_loss = torch.tensor(0.0, device=mlm_loss.device)
        if self.args.lam > 0:
            with torch.no_grad():
                target_lm_out = self.lm_encoder(
                    input_ids=input_ids,  # unmasked, per paper
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                target_lm_hidden = target_lm_out.last_hidden_state
                target_node_features = target_lm_hidden[:, 0, :]

                target_graph_embeddings = self.target_gnn_encoder(
                    target_node_features, edge_index
                )
                target_combined = self._fuse(
                    target_lm_hidden,
                    target_graph_embeddings,
                    self.target_fuse_lm,
                    self.target_fuse_gnn,
                    self.target_fuse_norm,
                )
                target_embeddings = self.target_projector(target_combined).detach()

            online_embeddings = self.projector(combined)

            # Paper Eq. 5 uses cosine similarity, not MSE.
            latent_loss = (
                1.0
                - F.cosine_similarity(
                    online_embeddings, target_embeddings, dim=-1
                ).mean()
            )

            self._update_target_networks(tau=getattr(self.args, 'ema_tau', 0.996))

        total_loss = mlm_loss + self.args.lam * latent_loss
        return total_loss, latent_loss

    def _mask_token_id(self) -> int:
        # Cached on first access to avoid attribute lookups in hot loop.
        if not hasattr(self, '_cached_mask_id'):
            # Heuristic: BERT/DeBERTa tokenizers use 103 for [MASK]; pass via args
            # if you use a different tokenizer.
            self._cached_mask_id = getattr(self.args, 'mask_token_id', 103)
        return self._cached_mask_id

    @torch.no_grad()
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        edge_index: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        lm_out = self.lm_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        lm_hidden = lm_out.last_hidden_state
        node_features = lm_hidden[:, 0, :]
        graph_embeddings = self.gnn_encoder(node_features, edge_index)
        combined = self._fuse(
            lm_hidden,
            graph_embeddings,
            self.fuse_lm,
            self.fuse_gnn,
            self.fuse_norm,
        )
        return combined
