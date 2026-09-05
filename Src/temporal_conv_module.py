```python
"""Temporal context encoder for spatio-temporal forecasting.

This module is intentionally separate from forecasting and graph decoding.
It models the temporal evolution of already fused node embeddings.

Expected input:
    [B, T, N, C]

where:
    B = batch size
    T = historical time length
    N = number of stations/nodes
    C = embedding dimension

Each station is processed independently along the temporal dimension.
The encoder uses residual temporal blocks with dilated 1D convolutions
followed by stride-2 average-pooling downsampling to progressively
compress the historical sequence into a shorter latent trajectory.
"""

from __future__ import annotations

from typing import Optional, List

import torch
import torch.nn as nn


class TemporalResidualBlock(nn.Module):
    """Residual temporal block with dilated Conv1D layers and
    stride-2 average-pooling downsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

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

        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.downsample = nn.AvgPool1d(
            kernel_size=2,
            stride=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual temporal processing and downsampling.

        Args:
            x: [B*N, C, T]

        Returns:
            [B*N, C_out, T_out], where T_out is approximately T/2.
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected input [B*N, C, T], got {tuple(x.shape)}"
            )

        residual = self.residual(x)

        y = self.conv1(x)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.drop1(y)

        y = self.conv2(y)
        y = self.norm2(y)
        y = self.act2(y)
        y = self.drop2(y)

        y = y + residual
        y = self.downsample(y)

        return y


class TemporalContextEncoder(nn.Module):
    """Encode and compress historical temporal context.

    Input:
        [B, T, N, C]

    Output:
        [B, W, N, C_out]

    The temporal dimension is progressively compressed by approximately
    a factor of two after each residual temporal block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dilation_list: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")

        self.in_channels = in_channels
        self.out_channels = (
            in_channels if out_channels is None else out_channels
        )
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.dropout = dropout

        if dilation_list is None:
            dilation_list = [1, 2, 4, 8]

        if len(dilation_list) < num_blocks:
            raise ValueError(
                "dilation_list must contain at least num_blocks entries"
            )

        self.dilation_list = dilation_list

        self.blocks = nn.ModuleList()

        for idx in range(num_blocks):
            block_in_channels = (
                in_channels if idx == 0 else self.out_channels
            )

            self.blocks.append(
                TemporalResidualBlock(
                    in_channels=block_in_channels,
                    out_channels=self.out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation_list[idx],
                    dropout=dropout,
                )
            )

    @staticmethod
    def _reshape_for_temporal_conv(
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Convert [B, T, N, C] to [B*N, C, T]."""

        if x.dim() != 4:
            raise ValueError(
                f"Expected input [B, T, N, C], got {tuple(x.shape)}"
            )

        b, t, n, c = x.shape

        return (
            x.permute(0, 2, 3, 1)
            .reshape(b * n, c, t)
        )

    @staticmethod
    def _reshape_back(
        x: torch.Tensor,
        batch_size: int,
        num_nodes: int,
        time_len: int,
    ) -> torch.Tensor:
        """Convert [B*N, C, T] back to [B, T, N, C]."""

        return (
            x.reshape(
                batch_size,
                num_nodes,
                -1,
                time_len,
            )
            .permute(0, 3, 1, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode historical temporal context.

        Args:
            x: [B, T, N, C]

        Returns:
            [B, W, N, C_out]
        """

        if x.dim() != 4:
            raise ValueError(
                f"Expected input [B, T, N, C], got {tuple(x.shape)}"
            )

        batch_size, time_len, num_nodes, channels = x.shape

        if channels != self.in_channels:
            raise ValueError(
                f"Expected input feature dimension C={self.in_channels}, "
                f"got C={channels}"
            )

        x_flat = self._reshape_for_temporal_conv(x)

        for block in self.blocks:

            # AvgPool1d(kernel_size=2, stride=2) requires T >= 2.
            if time_len < 2:
                break

            x_flat = block(x_flat)

            # AvgPool1d with kernel_size=2 and stride=2
            # produces floor(T / 2) for the current setting.
            time_len = time_len // 2

        return self._reshape_back(
            x_flat,
            batch_size,
            num_nodes,
            time_len,
        )


if __name__ == "__main__":

    B, T, N, C = 4, 336, 20, 8

    x = torch.randn(B, T, N, C)

    encoder = TemporalContextEncoder(
        in_channels=C,
        out_channels=8,
        num_blocks=4,
        kernel_size=3,
        dilation_list=[1, 2, 4, 8],
        dropout=0.1,
    )

    out = encoder(x)

    print("Input shape: ", x.shape)
    print("Output shape:", out.shape)
```
