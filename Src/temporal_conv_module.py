"""Temporal context encoder for spatio-temporal forecasting.

This module is intentionally separate from forecasting and graph decoding.
It only models the temporal evolution of already fused node embeddings.

The expected input is [B, T, N, C], where:
- B is batch size,
- T is the historical time length,
- N is the number of stations/nodes,
- C is the embedding dimension.

Each station is processed independently along the temporal dimension.
The implementation uses residual temporal blocks with dilated 1D convolutions
and stride-2 downsampling so the sequence length is progressively compressed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class TemporalResidualBlock(nn.Module):
    """A residual temporal block with two dilated Conv1D layers and stride-2 downsampling."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding="same",
            dilation=dilation,
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding="same",
            dilation=dilation,
        )
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal convolution and downsample along the sequence axis.

        Args:
            x: [B, T, N, C] or [B * N, C, T]

        Returns:
            [B, T // 2, N, C_out] when input is [B, T, N, C],
            otherwise [B * N, C_out, T // 2].
        """
        if x.dim() == 4:
            batch_size, time_len, num_nodes, channels = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, channels, time_len)
            is_batched = True
        elif x.dim() == 3:
            x_flat = x
            is_batched = False
        else:
            raise ValueError(f"Expected input [B, T, N, C] or [B * N, C, T], got {tuple(x.shape)}")

        residual = self.residual(x_flat)

        y = self.conv1(x_flat)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.drop1(y)

        y = self.conv2(y)
        y = self.norm2(y)
        y = self.act2(y)
        y = self.drop2(y)

        y = y + residual
        y = self.downsample(y)

        if is_batched:
            return y.reshape(batch_size, num_nodes, self.out_channels, y.shape[-1]).permute(0, 3, 1, 2)

        return y


class TemporalContextEncoder(nn.Module):
    """Stack of temporal residual blocks that compress the time dimension.

    The module preserves the station axis and only applies temporal convolutions.
    It is suitable for turning [B, T, N, C] into a compressed latent trajectory
    [B, W, N, C] where W << T.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dilation_list: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.dilation_list = dilation_list or [1, 2, 4, 8]
        self.dropout = dropout

        if len(self.dilation_list) < num_blocks:
            raise ValueError("dilation_list must contain at least num_blocks entries")

        self.blocks = nn.ModuleList()
        channels = in_channels
        for idx in range(num_blocks):
            block_out_channels = self.out_channels if idx == num_blocks - 1 else self.out_channels
            self.blocks.append(
                TemporalResidualBlock(
                    in_channels=channels,
                    out_channels=block_out_channels,
                    kernel_size=kernel_size,
                    dilation=self.dilation_list[idx],
                    dropout=dropout,
                )
            )
            channels = block_out_channels

    def _reshape_for_temporal_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [B, T, N, C] to [B * N, C, T] for Conv1D."""
        if x.dim() != 4:
            raise ValueError(f"Expected input [B, T, N, C], got {tuple(x.shape)}")

        b, t, n, c = x.shape
        return x.permute(0, 2, 3, 1).reshape(b * n, c, t)

    def _reshape_back(self, x: torch.Tensor, batch_size: int, num_nodes: int, time_len: int) -> torch.Tensor:
        """Convert [B * N, C, T] back to [B, T, N, C]."""
        return x.reshape(batch_size, num_nodes, -1, time_len).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode temporal context while downsampling the sequence length.

        Args:
            x: [B, T, N, C]

        Returns:
            [B, W, N, C_out], where W is progressively reduced by factor 2 per block.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected input [B, T, N, C], got {tuple(x.shape)}")

        batch_size, time_len, num_nodes, channels = x.shape
        x_flat = self._reshape_for_temporal_conv(x)

        for block in self.blocks:
            x_flat = block(x_flat)
            time_len = time_len // 2
            if time_len < 1:
                break

        return self._reshape_back(x_flat, batch_size, num_nodes, time_len)


if __name__ == '__main__':
    B, T, N, C = 4, 336, 20, 8
    x = torch.randn(B, T, N, C)
    encoder = TemporalContextEncoder(in_channels=C, out_channels=8, num_blocks=4, kernel_size=3, dropout=0.1)
    out = encoder(x)
    print('Output shape:', out.shape)
