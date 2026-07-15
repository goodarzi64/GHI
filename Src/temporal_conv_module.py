"""
Temporal convolution module for post-spatial temporal encoding.

The module expects an input tensor of shape [B, W, N, F_in], where:
- B is the batch size,
- W is the temporal length,
- N is the number of nodes,
- F_in is the feature dimension produced by the spatial encoder.

It applies a stack of 1D temporal convolutions along the W dimension using
exponentially increasing dilation rates. The temporal length is preserved, and
the output has shape [B, W, N, D].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class CausalResidualBlock(nn.Module):
    """A two-layer residual TCN block with causal convolutions."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=0, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def _apply_causal_padding(self, x: torch.Tensor) -> torch.Tensor:
        left_pad = self.dilation * (self.kernel_size - 1)
        return F.pad(x, (left_pad, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        y = self._apply_causal_padding(x)
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.drop1(y)

        y = self._apply_causal_padding(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.act2(y)
        y = self.drop2(y)

        return y + residual


class TemporalConvModule(nn.Module):
    """Apply a causal TCN with residual blocks over node features."""

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        dilation_list: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.dilation_list = dilation_list or [1, 2, 4, 8]
        self.num_layers = len(self.dilation_list)
        self.kernel_size = kernel_size

        self.temporal_layers = nn.ModuleList()
        in_channels = node_feature_dim
        for dilation in self.dilation_list:
            self.temporal_layers.append(
                CausalResidualBlock(in_channels, self.hidden_dim, self.kernel_size, dilation, dropout)
            )
            in_channels = self.hidden_dim


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process [B, W, N, F_in] into [B, W, N, D]."""
        if x.dim() != 4:
            raise ValueError(f"Expected input [B, W, N, F_in], got {x.shape}")

        B, W, N, C = x.shape

        # Flatten nodes so temporal convolution is applied independently per node.
        x_flat = x.permute(0, 2, 3, 1).reshape(B * N, C, W)

        for layer in self.temporal_layers:
            x_flat = layer(x_flat)

        # Restore [B, W, N, hidden_dim].
        x_flat = x_flat.reshape(B, N, self.hidden_dim, W).permute(0, 3, 1, 2)

        # Return node-time feature vectors with hidden dimension.
        x_flat = x_flat.reshape(B, W, N, self.hidden_dim)
        return x_flat


if __name__ == '__main__':
    B, W, N, F_in = 4, 48, 20, 8
    x = torch.randn(B, W, N, F_in)
    module = TemporalConvModule(
        node_feature_dim=F_in,
        hidden_dim=64,
        dilation_list=[1, 2, 4, 8],
        kernel_size=3,
        dropout=0.1,
    )
    out = module(x)
    print('Output shape:', out.shape)
