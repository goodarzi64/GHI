
"""MTRL (Multi scale Temporal Representation Learning).

This module is intentionally separate from forecasting and graph decoding.
It models the temporal evolution of already fused node embeddings.

Expected input:
    [B, T, N, C]

where:
    B = batch size
    T = historical time length
    N = number of stations/nodes
    C = embedding dimension

Each station is processed independently along the temporal dimension. Each
residual block uses left-padded causal dilated 1D convolutions: an output at
time t can depend only on inputs at t and earlier times. Per-time-step channel
normalization avoids aggregating statistics across future time positions.
After the causal convolutions and residual addition, stride-2 average pooling
compresses the historical sequence into a shorter latent trajectory. Each
pooled output summarizes its pair of input steps and is aligned to the later
step in that pair.
"""

from __future__ import annotations

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalResidualBlock(nn.Module):
    """Causal dilated residual block with stride-2 temporal downsampling.

    Both convolutions are padded only on the left by
    ``dilation * (kernel_size - 1)``, so the output at each time position
    cannot use values from later positions. Normalization acts across channels
    independently at each time step. The residual branch uses a pointwise
    convolution when the channel dimensions differ. After residual addition,
    average pooling combines adjacent positions; each pooled value is aligned
    with the later position in its pair.
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

        # Left-only padding preserves sequence length without looking ahead.
        self.left_padding = dilation * (kernel_size - 1)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
        )
        # LayerNorm is applied after transposing to [batch, time, channels],
        # so each time step is normalized independently across channels.
        self.norm1 = nn.LayerNorm(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
        )
        self.norm2 = nn.LayerNorm(out_channels)
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
            x: [B, T, N, C] or [B*N, C, T]

        Returns:
            [B, T_out, N, C_out] or [B*N, C_out, T_out], depending on input form.
        """
        if x.dim() == 4:
            b, t, n, c = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(b * n, c, t)
            out_flat = self.forward(x_flat)
            out = out_flat.reshape(b, n, self.out_channels, -1).permute(0, 3, 1, 2)
            return out

        if x.dim() != 3:
            raise ValueError(
                f"Expected input [B*N, C, T] or [B, T, N, C], got {tuple(x.shape)}"
            )

        residual = self.residual(x)

        y = self.conv1(F.pad(x, (self.left_padding, 0)))
        y = self.norm1(y.transpose(1, 2)).transpose(1, 2)
        y = self.act1(y)
        y = self.drop1(y)

        y = self.conv2(F.pad(y, (self.left_padding, 0)))
        y = self.norm2(y.transpose(1, 2)).transpose(1, 2)
        y = self.act2(y)
        y = self.drop2(y)

        y = y + residual
        y = self.downsample(y)

        return y


class TemporalContextEncoder(nn.Module):
    """Causally encode and compress historical temporal context.

    Input:
        [B, T, N, C]

    Output:
        [B, W, N, 2 * C_out]

    Each block applies causal dilated convolutions, so no temporal position
    uses information from later positions. The temporal dimension is then
    compressed by approximately a factor of two through average pooling.
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
        self.output_channels = 2 * self.out_channels
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

        # Applied independently at every remaining time position and node.
        self.short_sequence_projection = (
            nn.Linear(in_channels, self.out_channels)
            if in_channels != self.out_channels
            else nn.Identity()
        )
        self.output_projection = nn.Linear(self.out_channels, self.output_channels)

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
        """Encode historical temporal context with no convolutional lookahead.

        Args:
            x: [B, T, N, C]

        Returns:
            [B, W, N, 2 * C_out], where each position depends only on its
            corresponding and earlier input positions. The final linear layer
            acts on channels independently at each node and time position.
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

        x_out = self._reshape_back(
            x_flat,
            batch_size,
            num_nodes,
            time_len,
        )
        if x_out.shape[-1] != self.out_channels:
            # If the input is shorter than one pooling window, no residual
            # block runs, so perform its channel adjustment directly.
            x_out = self.short_sequence_projection(x_out)
        return self.output_projection(x_out)


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

