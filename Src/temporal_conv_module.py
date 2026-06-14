"""
Temporal Convolution Module for Diurnal and Seasonal Patterns

This module applies separate temporal convolutions for diurnal and seasonal
patterns and fuses them into a unified node-level representation.

It expects fused spatial features over time and does not operate on separate
spatial graph branches. Spatial branch fusion should be completed before
calling this module.

Expected input shape for the forward pass is [B, W, N, F_in]
where B=batch size, W=window length, N=number of nodes, F_in=node features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class TemporalBranch(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=((kernel_size - 1) // 2) * dilation,
            dilation=dilation,
        )
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal convolution to [B*N, F_in, W] and pool over time."""
        x = self.conv(x)
        x = self.activation(x)
        x = x.mean(dim=-1)
        x = self.norm(x)
        return x


class TemporalConvModule(nn.Module):
    """Apply diurnal, seasonal, and optional global temporal filters to fused spatial features.

    The module expects a fused spatial tensor [B, W, N, F_in] as input. It does not
    perform fusion across spatial graph branches; that should be done prior to calling
    this module.

    Diurnal, seasonal, and optional global filters are combined into a single temporal
    representation, producing one unified output per node.
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        output_dim: Optional[int] = None,
        diurnal_window: int = 24,
        seasonal_window: int = 168,
        seasonal_dilation: int = 1,
        dropout: float = 0.1,
        use_global_branch: bool = True,
        merge_method: str = 'sum',
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or hidden_dim
        self.use_global_branch = use_global_branch
        self.merge_method = merge_method

        self.diurnal_branch = TemporalBranch(
            in_channels=node_feature_dim,
            out_channels=hidden_dim,
            kernel_size=diurnal_window,
            dilation=1,
        )

        self.seasonal_branch = TemporalBranch(
            in_channels=node_feature_dim,
            out_channels=hidden_dim,
            kernel_size=seasonal_window,
            dilation=seasonal_dilation,
        )

        if use_global_branch:
            self.global_branch = TemporalBranch(
                in_channels=node_feature_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                dilation=1,
            )
        else:
            self.global_branch = None

        if merge_method not in {'sum', 'mean', 'concat'}:
            raise ValueError("merge_method must be one of 'sum', 'mean', or 'concat'")

        proj_input_dim = hidden_dim * (3 if self.global_branch is not None else 2) if merge_method == 'concat' else hidden_dim
        self.project = nn.Sequential(
            nn.Linear(proj_input_dim, self.output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.output_dim, self.output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Fused spatial sequence tensor of shape [B, W, N, F_in]
        Returns:
            fused: [B, N, output_dim]

        Note:
            This module assumes the spatial fusion of graph branches has already
            been performed, so it does not receive separate spatial branch inputs.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected input [B, W, N, F_in], got {x.shape}")

        B, W, N, F_in = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * N, F_in, W)

        diurnal_out = self.diurnal_branch(x)
        seasonal_out = self.seasonal_branch(x)
        outputs = [diurnal_out, seasonal_out]

        if self.global_branch is not None:
            global_out = self.global_branch(x)
            outputs.append(global_out)

        if self.merge_method == 'sum':
            fused = sum(outputs)
        elif self.merge_method == 'mean':
            fused = sum(outputs) / len(outputs)
        else:  # concat
            fused = torch.cat(outputs, dim=-1)

        fused = self.project(fused)
        fused = fused.reshape(B, N, -1)
        return fused


if __name__ == '__main__':
    B, W, N, F_in = 4, 48, 20, 8
    x = torch.randn(B, W, N, F_in)
    module = TemporalConvModule(
        node_feature_dim=F_in,
        hidden_dim=64,
        output_dim=64,
        diurnal_window=24,
        seasonal_window=168,
        seasonal_dilation=1,
        dropout=0.1,
        merge_method='sum',
    )
    out = module(x)
    print('Output shape:', out.shape)
