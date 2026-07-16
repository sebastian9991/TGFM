from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv, GCNConv


def create_activation(name: str) -> nn.Module:
    if name == 'relu':
        return nn.ReLU()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'prelu':
        return nn.PReLU()
    elif name == 'elu':
        return nn.ELU()
    elif name == 'none':
        return nn.Identity()
    else:
        raise NotImplementedError(f'{name} is not implemented.')


class GATResidualBlock(nn.Module):
    """A single GAT block that can optionally include.
    - Normalization (batch or layer)
    - Residual connection (with dimension matching)
    - Activation function
    - Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        attn_drop: float = 0.1,
        feat_drop: float = 0.1,
        norm: str = 'none',  # 'none', 'batch', or 'layer'
        activation: str = 'prelu',  # You can pass None if you don't want activation
        use_residual: bool = False,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.dropout = feat_drop
        self.activation = create_activation(activation)

        self.conv = GATConv(
            in_channels,
            out_channels,
            heads=heads,
            dropout=attn_drop,
            concat=(heads > 1),
        )

        # Determine output dimension after GATConv
        self.out_dim = out_channels * heads if heads > 1 else out_channels
        self.norm: Optional[Union[nn.BatchNorm1d, nn.LayerNorm]] = None

        # Define optional normalization
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(self.out_dim)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(self.out_dim)
        else:
            self.norm = None

        self.residual_transform: Optional[nn.Linear] = None
        # 3. Match dimensions (if in_channels != out_dim)
        if self.use_residual and (in_channels != self.out_dim):
            self.residual_transform = nn.Linear(in_channels, self.out_dim)
        else:
            self.residual_transform = None

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        identity = x

        out = self.conv(x, edge_index)

        if self.norm is not None:
            out = self.norm(out)

        if self.activation is not None:
            out = self.activation(out)

        out = F.dropout(out, p=self.dropout, training=self.training)

        if self.use_residual:
            if self.residual_transform is not None:
                identity = self.residual_transform(identity)
            out = out + identity

        return out

    def reset_parameters(self) -> None:
        """Re-initialize learnable parameters.
        - GATConv
        - Normalization
        - Residual transform
        """
        self.conv.reset_parameters()

        if self.norm is not None:
            # Both BatchNorm1d and LayerNorm have reset_parameters()
            self.norm.reset_parameters()

        if self.residual_transform is not None:
            self.residual_transform.reset_parameters()

        if isinstance(self.activation, nn.PReLU):
            self.activation.weight.data.fill_(0.25)  # Default PReLU initialization


class GATNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        attn_drop: float = 0.1,
        feat_drop: float = 0.1,
        heads: int = 1,
        norm: str = 'none',
        activation: str = 'prelu',
        use_residual: bool = False,
    ):
        super().__init__()
        assert (
            num_layers >= 2
        ), 'num_layers should be at least 2 for (input -> hidden -> output).'

        self.num_layers = num_layers
        self.dropout = feat_drop

        # Create a list of blocks
        self.blocks = nn.ModuleList()

        self.blocks.append(
            GATResidualBlock(
                in_channels=in_channels,
                out_channels=hidden_channels,
                heads=heads,
                attn_drop=attn_drop,
                feat_drop=0,
                norm=norm,
                activation=activation,
                use_residual=use_residual,
            )
        )

        for _ in range(num_layers - 2):
            self.blocks.append(
                GATResidualBlock(
                    in_channels=hidden_channels * heads,
                    out_channels=hidden_channels,
                    heads=heads,
                    attn_drop=attn_drop,
                    feat_drop=0,
                    norm=norm,
                    activation=activation,
                    use_residual=use_residual,
                )
            )

        self.blocks.append(
            GATResidualBlock(
                in_channels=hidden_channels * heads,
                out_channels=out_channels,
                heads=1,
                attn_drop=attn_drop,
                feat_drop=0,
                norm='none',
                activation='none',
                use_residual=use_residual,
            )
        )

        self.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for block in self.blocks:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = block(x, edge_index)
        return x

    def reset_parameters(self) -> None:
        """Re-initialize all parameters in each block"""
        for block in self.blocks:
            block.reset_parameters()

    def get_trainable_parameters(
        self, n_trainable_layers: int
    ) -> List[torch.nn.Parameter]:
        if n_trainable_layers != -1:
            raise NotImplementedError(
                'n_trainable_layers is not implemented for GATNet.'
            )
        return list(self.parameters())


class GCNResidualBlock(nn.Module):
    """A single GCN block that can optionally include.
    - Normalization (batch or layer)
    - Residual connection (with dimension matching)
    - Activation function
    - Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1,
        norm: str = 'none',  # 'none', 'batch', or 'layer'
        activation: str = 'relu',
        use_residual: bool = False,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.dropout = dropout
        self.activation = create_activation(activation)

        # GCNConv layer
        self.conv = GCNConv(
            in_channels, out_channels, add_self_loops=True, normalize=True
        )

        # Normalization (optional)
        self.norm: Optional[Union[nn.BatchNorm1d, nn.LayerNorm]] = None
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = None

        # Residual connection (optional)
        self.residual_transform: Optional[nn.Linear] = None
        if self.use_residual and (in_channels != out_channels):
            self.residual_transform = nn.Linear(in_channels, out_channels)
        else:
            self.residual_transform = None

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        identity = x

        out = self.conv(x, edge_index)

        if self.norm is not None:
            out = self.norm(out)

        out = self.activation(out)

        out = F.dropout(out, p=self.dropout, training=self.training)

        # Residual connection
        if self.use_residual:
            if self.residual_transform is not None:
                identity = self.residual_transform(identity)
            out = out + identity

        return out

    def reset_parameters(self) -> None:
        """Re-initialize learnable parameters.
        - GCNConv
        - Normalization
        - Residual transform
        """
        self.conv.reset_parameters()
        if self.norm is not None:
            self.norm.reset_parameters()
        if self.residual_transform is not None:
            self.residual_transform.reset_parameters()
        if isinstance(self.activation, nn.PReLU):
            self.activation.weight.data.fill_(0.25)  # Default PReLU initialization


class GCNNet(nn.Module):
    """A multi-layer GCN that stacks GCNResidualBlock blocks.
    The final layer outputs out_channels features.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.1,
        norm: str = 'none',
        activation: str = 'relu',
        use_residual: bool = False,
    ):
        super().__init__()
        assert (
            num_layers >= 2
        ), 'num_layers should be at least 2 (input -> hidden -> output).'
        self.num_layers = num_layers

        # Build blocks
        self.blocks = nn.ModuleList()

        self.blocks.append(
            GCNResidualBlock(
                in_channels=in_channels,
                out_channels=hidden_channels,
                dropout=dropout,  # block itself handles dropout
                norm=norm,
                activation=activation,
                use_residual=use_residual,
            )
        )

        for _ in range(num_layers - 2):
            self.blocks.append(
                GCNResidualBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    dropout=dropout,
                    norm=norm,
                    activation=activation,
                    use_residual=use_residual,
                )
            )

        self.blocks.append(
            GCNResidualBlock(
                in_channels=hidden_channels,
                out_channels=out_channels,
                dropout=0.0,
                norm='none',
                activation='none',
                use_residual=use_residual,
            )
        )

        self.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x, edge_index)
        return x

    def reset_parameters(self) -> None:
        """Re-initialize all parameters in each block."""
        for block in self.blocks:
            block.reset_parameters()

    def get_trainable_parameters(
        self, n_trainable_layers: int
    ) -> List[torch.nn.Parameter]:
        if n_trainable_layers != -1:
            raise NotImplementedError(
                'n_trainable_layers is not implemented for GCNNet.'
            )
        return list(self.parameters())
