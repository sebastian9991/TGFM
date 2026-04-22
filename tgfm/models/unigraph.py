"""Implmentation of Unigraph
@article{he2025unigraph,
  title={UniGraph: Learning a Unified Cross-Domain Foundation Model for Text-Attributed Graphs},
  author={He, Yufei and Sui, Yuan and He, Xiaoxin and Hooi, Bryan},
  journal={ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year={2025}
}.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from tgfm.models.gat import GAT
from tgfm.utils.args import UnigraphArguments
from tgfm.utils.loss import compute_mlm_loss


class UniGraph(nn.Module):
    """UniGraph: Learning a Unified Cross-Domain Foundation Model for Text-Attributed Graphs."""

    def __init__(self, args: UnigraphArguments):
        super().__init__()
        self.args = args

        # Language model encoder
        self.lm_encoder = AutoModel.from_pretrained(args.lm_type)

        # GNN encoder
        self.gnn_encoder = GAT(
            in_dim=args.hidden_size,
            hidden_dim=args.hidden_size,
            out_dim=args.hidden_size,
            num_layers=args.num_layers,
            nhead=args.nhead,
            feat_drop=args.dropout,
        )

        # Fusion layer to combine LM and GNN outputs
        self.fusion = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.LayerNorm(args.hidden_size),
            nn.ReLU(),
        )

        # Projector for latent space regularization
        if args.lam > 0:
            self.projector = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.LayerNorm(args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.hidden_size),
            )

            # Target networks for latent space regularization
            self.target_lm_encoder = AutoModel.from_pretrained(args.lm_type)
            self.target_gnn_encoder = GAT(
                in_dim=args.hidden_size,
                hidden_dim=args.hidden_size,
                out_dim=args.hidden_size,
                num_layers=args.num_layers,
                nhead=args.nhead,
                feat_drop=args.dropout,
            )
            self.target_fusion = nn.Sequential(
                nn.Linear(args.hidden_size * 2, args.hidden_size),
                nn.LayerNorm(args.hidden_size),
                nn.ReLU(),
            )
            self.target_projector = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.LayerNorm(args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.hidden_size),
            )

            # Initialize target networks
            self._init_target_networks()

        # MLM head
        self.mlm_head = nn.Linear(args.hidden_size, self.lm_encoder.config.vocab_size)

    def _init_target_networks(self) -> None:
        """Initialize target networks with the same weights as online networks."""
        self.target_lm_encoder.load_state_dict(self.lm_encoder.state_dict())
        self.target_gnn_encoder.load_state_dict(self.gnn_encoder.state_dict())
        self.target_fusion.load_state_dict(self.fusion.state_dict())
        self.target_projector.load_state_dict(self.projector.state_dict())

        # Freeze target networks
        for param in self.target_lm_encoder.parameters():
            param.requires_grad = False
        for param in self.target_gnn_encoder.parameters():
            param.requires_grad = False
        for param in self.target_fusion.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _update_target_networks(self, tau: float = 0.99) -> None:
        """Update target networks using exponential moving average."""
        for target_param, param in zip(
            self.target_lm_encoder.parameters(), self.lm_encoder.parameters()
        ):
            target_param.data = tau * target_param.data + (1 - tau) * param.data
        for target_param, param in zip(
            self.target_gnn_encoder.parameters(), self.gnn_encoder.parameters()
        ):
            target_param.data = tau * target_param.data + (1 - tau) * param.data
        for target_param, param in zip(
            self.target_fusion.parameters(), self.fusion.parameters()
        ):
            target_param.data = tau * target_param.data + (1 - tau) * param.data
        for target_param, param in zip(
            self.target_projector.parameters(), self.projector.parameters()
        ):
            target_param.data = tau * target_param.data + (1 - tau) * param.data

    def forward(
        self,
        input_ids: torch.Tensor,
        masked_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model."""
        # Get node features from language model
        lm_outputs = self.lm_encoder(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # TODO Check the dimensions on this
        node_features = lm_outputs.last_hidden_state[
            :, 0
        ]  # [CLS] token as initalization to node features.

        # Get graph embeddings from GNN
        graph_embeddings = self.gnn_encoder(edge_index, node_features)

        # Combine LM and GNN outputs
        combined = self.fusion(torch.cat([node_features, graph_embeddings], dim=-1))

        # Compute MLM loss
        mlm_logits = self.mlm_head(combined)
        mlm_loss = compute_mlm_loss(mlm_logits, input_ids, masked_input_ids)

        # Initialize latent loss
        latent_loss = torch.tensor(0.0, device=mlm_loss.device)

        # Compute latent space regularization loss if enabled
        if self.args.lam > 0:
            # Get target embeddings
            with torch.no_grad():
                target_lm_outputs = self.target_lm_encoder(
                    input_ids=masked_input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                target_node_features = target_lm_outputs.last_hidden_state[:, 0]
                target_graph_embeddings = self.target_gnn_encoder(
                    edge_index, target_node_features
                )
                target_combined = self.target_fusion(
                    torch.cat([target_node_features, target_graph_embeddings], dim=-1)
                )
                target_embeddings = self.target_projector(target_combined)

            # Get online embeddings
            online_embeddings = self.projector(combined)

            # Compute latent loss
            latent_loss = F.mse_loss(online_embeddings, target_embeddings)

            # Update target networks
            self._update_target_networks()

        # Combine losses
        total_loss = mlm_loss + self.args.lam * latent_loss

        return total_loss, latent_loss

    def get_embeddings(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Get node embeddings for inference."""
        with torch.no_grad():
            # Get inputs
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            token_type_ids = batch['token_type_ids']
            graph = batch['graph']

            # Get node features from language model
            lm_outputs = self.lm_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            node_features = lm_outputs.last_hidden_state[:, 0]

            # Get graph embeddings from GNN
            graph_embeddings = self.gnn_encoder(graph, node_features)

            # Combine LM and GNN outputs
            combined = self.fusion(torch.cat([node_features, graph_embeddings], dim=-1))

            return combined
