"""Latent Graph Evolution Encoder (LGEE).

This module converts a compressed historical latent trajectory into horizon-specific
latent graph evolution contexts. It is intentionally not a forecasting head and
does not predict node values or adjacency matrices.

The module performs node-wise cross-attention between learnable horizon queries
and historical latent states along the temporal dimension. Each forecasting
horizon learns which historical latent tokens are most informative for its own
context.

Input
-----
Z_hist: [B, W, N, C]
    Historical latent trajectory, where:
    - B: batch size
    - W: number of historical temporal tokens
    - N: number of graph nodes/stations
    - C: latent feature dimension

Output
------
Z_graph: [B, H, N, C]
    Horizon-conditioned latent graph evolution contexts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):
    """Two-layer feed-forward network used in each transformer block."""

    def __init__(self, latent_dim: int, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = latent_dim * 2 if hidden_dim is None else hidden_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadCrossAttention(nn.Module):
    """Vectorized multi-head cross-attention over the temporal dimension.

    Queries come from horizon embeddings and keys/values come from the historical
    latent trajectory. The operation is applied independently per node.
    """

    def __init__(self, latent_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads")

        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape [B * N, L, C] to [B * N, H, L, d]."""
        bnl, length, channels = x.shape
        return x.view(bnl, length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Apply multi-head cross-attention.

        Args:
            queries: [B * N, Hq, C]
            keys: [B * N, W, C]
            values: [B * N, W, C]

        Returns:
            [B * N, Hq, C]
        """
        q = self.q_proj(queries)
        k = self.k_proj(keys)
        v = self.v_proj(values)

        q = self._reshape_for_attention(q)
        k = self._reshape_for_attention(k)
        v = self._reshape_for_attention(v)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(queries.shape[0], queries.shape[1], self.latent_dim)
        return self.out_proj(context)


class HorizonAwareCrossAttentionBlock(nn.Module):
    """A single transformer-style cross-attention block."""

    def __init__(self, latent_dim: int, num_heads: int, dropout: float = 0.1, ff_hidden_dim: int | None = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.attn = MultiHeadCrossAttention(latent_dim, num_heads, dropout=dropout)
        self.ffn = FeedForwardNetwork(latent_dim, hidden_dim=ff_hidden_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Perform cross-attention and feed-forward refinement."""
        attn_out = self.attn(self.norm1(queries), self.norm1(keys), self.norm1(values))
        attn_out = queries + self.dropout(attn_out)

        ffn_out = self.ffn(self.norm2(attn_out))
        return attn_out + self.dropout(ffn_out)


class LatentGraphEvolutionEncoder(nn.Module):
    """Learn horizon-conditioned latent graph evolution contexts.

    The encoder takes a compressed historical latent trajectory [B, W, N, C] and
    produces a horizon-specific latent graph representation [B, H, N, C]. Each
    forecasting horizon has its own learnable query embedding, enabling the model
    to attend to different parts of the historical trajectory.
    """

    def __init__(
        self,
        latent_dim: int,
        num_horizons: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        ff_hidden_dim: int | None = None,
        num_blocks: int = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_horizons = num_horizons
        self.num_heads = num_heads
        self.dropout = dropout

        self.horizon_embeddings = nn.Parameter(torch.randn(num_horizons, latent_dim))
        self.blocks = nn.ModuleList(
            [
                HorizonAwareCrossAttentionBlock(
                    latent_dim=latent_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    ff_hidden_dim=ff_hidden_dim,
                )
                for _ in range(num_blocks)
            ]
        )

    def _prepare_inputs(self, z_hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert [B, W, N, C] to vectorized [B*N, W, C] and [B*N, H, C]."""
        if z_hist.dim() != 4:
            raise ValueError(f"Expected input [B, W, N, C], got {tuple(z_hist.shape)}")

        batch_size, time_steps, num_nodes, latent_dim = z_hist.shape
        if latent_dim != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {latent_dim}")

        z_flat = z_hist.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, time_steps, latent_dim)
        horizon_queries = self.horizon_embeddings.unsqueeze(0).expand(batch_size * num_nodes, -1, latent_dim)
        return z_flat, horizon_queries, (batch_size, num_nodes)

    def _reshape_output(self, x: torch.Tensor, batch_size: int, num_nodes: int) -> torch.Tensor:
        """Reshape [B*N, H, C] back to [B, H, N, C]."""
        return x.reshape(batch_size, num_nodes, self.num_horizons, self.latent_dim).permute(0, 2, 1, 3)

    def forward(self, z_hist: torch.Tensor) -> torch.Tensor:
        """Produce horizon-conditioned latent graph evolution contexts.

        Args:
            z_hist: [B, W, N, C]

        Returns:
            [B, H, N, C]
        """
        z_flat, queries, (batch_size, num_nodes) = self._prepare_inputs(z_hist)
        keys = z_flat
        values = z_flat

        for block in self.blocks:
            queries = block(queries, keys, values)

        return self._reshape_output(queries, batch_size, num_nodes)


if __name__ == '__main__':
    x = torch.randn(2, 16, 4, 8)
    encoder = LatentGraphEvolutionEncoder(latent_dim=8, num_horizons=3, num_heads=2, dropout=0.1)
    out = encoder(x)
    print(out.shape)
