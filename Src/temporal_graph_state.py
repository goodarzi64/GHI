import torch
import torch.nn as nn


class TemporalGraphStateAggregator(nn.Module):
    """Aggregate a historical latent sequence Z [B, W, N, C] into a compact state S [B, N, C]."""

    def __init__(self, channels: int, state_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.state_dim = channels if state_dim is None else state_dim
        self.query_proj = nn.Linear(channels, self.state_dim)
        self.key_proj = nn.Linear(channels, self.state_dim)
        self.value_proj = nn.Linear(channels, self.state_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 4:
            raise ValueError(f"Expected Z [B, W, N, C], got {tuple(z.shape)}")

        b, w, n, c = z.shape
        z_flat = z.permute(0, 2, 1, 3).reshape(b * n, w, c)

        query = self.query_proj(z_flat[:, -1, :])
        keys = self.key_proj(z_flat)
        values = self.value_proj(z_flat)

        scores = torch.einsum('bi,bwi->bw', query, keys) / (self.state_dim ** 0.5)
        alpha = torch.softmax(scores, dim=-1)
        alpha = self.dropout(alpha)

        state_flat = torch.einsum('bw,bwi->bi', alpha, values)
        return state_flat.reshape(b, n, self.state_dim)


class LatentGraphGenerator(nn.Module):
    """Predict a future graph increment from the compact latent graph state."""

    def __init__(self, state_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1 + 4 * state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, historical_adj: torch.Tensor, node_state: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if historical_adj.dim() != 3:
            raise ValueError(f"Expected historical adjacency [B, N, N], got {tuple(historical_adj.shape)}")
        if node_state.dim() != 3:
            raise ValueError(f"Expected node_state [B, N, C], got {tuple(node_state.shape)}")

        src_state = node_state[:, edge_index[0], :]
        dst_state = node_state[:, edge_index[1], :]
        current_edge = historical_adj[:, edge_index[0], edge_index[1]].unsqueeze(-1)
        diff_state = torch.abs(src_state - dst_state)
        hadamard_state = src_state * dst_state

        edge_features = torch.cat(
            [current_edge, src_state, dst_state, diff_state, hadamard_state],
            dim=-1,
        )
        delta = self.mlp(edge_features).squeeze(-1)
        return current_edge.squeeze(-1) + delta
