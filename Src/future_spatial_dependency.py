"""Utilities for generating future spatial dependency graphs.

This module applies the remaining forecast graph generation path used by the
pipeline: Z_hist [B, W, N, C] -> LGEE -> Z_graph [B, H, N, C] -> FSDG ->
Future Graphs -> FSDP.
"""

import torch
import torch.nn as nn


class FutureSpatialDependencyGenerator(nn.Module):
    """Predict horizon-aware future graphs from LGEE outputs with optional sparse Top-K outputs."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int | None = None,
        residual_scale: float = 0.1,
        k: int = 5,
        candidate_scale: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.residual_scale = residual_scale
        self.k = max(1, int(k))
        self.candidate_scale = max(1, int(candidate_scale))
        hidden_dim = 2 * latent_dim if hidden_dim is None else hidden_dim

        self.wind_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.sem_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, latent_dim),
        )

    def _candidate_size_for_horizon(self, num_horizons: int, horizon_idx: int) -> int:
        if num_horizons <= 1:
            return self.k
        farthest_k = max(self.k, self.k * self.candidate_scale)
        if horizon_idx <= 0:
            return self.k
        if horizon_idx >= num_horizons - 1:
            return farthest_k
        span = farthest_k - self.k
        step = span / max(num_horizons - 1, 1)
        return max(self.k, min(farthest_k, int(round(self.k + step * horizon_idx))))

    def _candidate_pool_for_horizon(self, current_adj: torch.Tensor, horizon_idx: int, num_horizons: int) -> torch.Tensor:
        if current_adj.dim() != 3:
            raise ValueError(f"Expected current adjacency [B, N, N], got {tuple(current_adj.shape)}")

        batch_size, num_nodes, _ = current_adj.shape
        candidate_k = self._candidate_size_for_horizon(num_horizons, horizon_idx)
        candidate_k = min(candidate_k, max(1, num_nodes - 1))

        candidate_pool = []
        for batch_idx in range(batch_size):
            row_pool = []
            for src_idx in range(num_nodes):
                row = current_adj[batch_idx, src_idx].clone()
                row[src_idx] = -torch.inf
                if row.numel() == 0:
                    row_pool.append(torch.empty((0,), device=current_adj.device, dtype=torch.long))
                    continue
                if row.abs().sum() == 0:
                    dst_idx = torch.arange(num_nodes, device=current_adj.device)
                    dst_idx = dst_idx[dst_idx != src_idx]
                    if dst_idx.numel() == 0:
                        row_pool.append(torch.empty((0,), device=current_adj.device, dtype=torch.long))
                    else:
                        row_pool.append(dst_idx[:candidate_k])
                    continue
                _, dst_idx = torch.topk(row, k=min(candidate_k, row.numel()), largest=True, sorted=True)
                row_pool.append(dst_idx)
            candidate_pool.append(torch.stack(row_pool, dim=0))
        return torch.stack(candidate_pool, dim=0)

    def _pairwise_features(self, z_graph_h: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        src = z_graph_h[:, candidates[:, 0], :]
        dst = z_graph_h[:, candidates[:, 1], :]
        return torch.cat([src, dst], dim=-1)

    def _dense_residual_horizon(self, z_h: torch.Tensor, current_adj: torch.Tensor, mlp: nn.Module, horizon_idx: int, num_horizons: int) -> torch.Tensor:
        batch_size, num_nodes, _ = z_h.shape
        candidate_pool = self._candidate_pool_for_horizon(current_adj, horizon_idx, num_horizons)
        out_adj = current_adj.clone()

        for batch_idx in range(batch_size):
            for src_idx in range(num_nodes):
                candidates = candidate_pool[batch_idx, src_idx]
                if candidates.numel() == 0:
                    continue
                candidates = candidates[candidates != src_idx]
                if candidates.numel() == 0:
                    continue
                src_state = z_h[batch_idx, src_idx].unsqueeze(0).expand(candidates.numel(), -1)
                dst_state = z_h[batch_idx, candidates]
                edge_features = torch.cat([src_state, dst_state], dim=-1)
                residuals = mlp(edge_features).mean(dim=-1) * self.residual_scale
                current_vals = current_adj[batch_idx, src_idx, candidates]
                updated = current_vals + residuals
                top_k = min(self.k, updated.numel())
                if top_k <= 0:
                    continue
                _, top_pos = torch.topk(updated, k=top_k, sorted=True)
                selected = candidates[top_pos]
                out_adj[batch_idx, src_idx, selected] = current_adj[batch_idx, src_idx, selected] + residuals[top_pos]
        return out_adj

    def _sparse_residual_horizon(self, z_h: torch.Tensor, current_adj: torch.Tensor, mlp: nn.Module, horizon_idx: int, num_horizons: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, _ = z_h.shape
        candidate_pool = self._candidate_pool_for_horizon(current_adj, horizon_idx, num_horizons)

        edge_index_blocks = []
        edge_weight_blocks = []

        for batch_idx in range(batch_size):
            for src_idx in range(num_nodes):
                candidates = candidate_pool[batch_idx, src_idx]
                if candidates.numel() == 0:
                    continue
                candidates = candidates[candidates != src_idx]
                if candidates.numel() == 0:
                    continue
                src_state = z_h[batch_idx, src_idx].unsqueeze(0).expand(candidates.numel(), -1)
                dst_state = z_h[batch_idx, candidates]
                edge_features = torch.cat([src_state, dst_state], dim=-1)
                residuals = mlp(edge_features).mean(dim=-1) * self.residual_scale
                current_vals = current_adj[batch_idx, src_idx, candidates]
                updated = current_vals + residuals
                top_k = min(self.k, updated.numel())
                if top_k <= 0:
                    continue
                _, top_pos = torch.topk(updated, k=top_k, sorted=True)
                selected = candidates[top_pos]
                selected_vals = current_adj[batch_idx, src_idx, selected] + residuals[top_pos]

                src_nodes = torch.full((selected.numel(),), src_idx, device=current_adj.device, dtype=torch.long) + batch_idx * num_nodes
                dst_nodes = selected + batch_idx * num_nodes
                edge_index_blocks.append(torch.stack([src_nodes, dst_nodes], dim=0))
                edge_weight_blocks.append(selected_vals)

        if not edge_index_blocks:
            return (
                torch.empty((2, 0), device=current_adj.device, dtype=torch.long),
                torch.empty((0,), device=current_adj.device, dtype=current_adj.dtype),
            )

        edge_index = torch.cat(edge_index_blocks, dim=1)
        edge_weight = torch.cat(edge_weight_blocks, dim=0)
        return edge_index, edge_weight

    def _predict_residuals(self, z_graph: torch.Tensor, current_adj: torch.Tensor, mlp: nn.Module) -> torch.Tensor:
        if z_graph.dim() != 4:
            raise ValueError(f"Expected z_graph [B, H, N, C], got {tuple(z_graph.shape)}")
        if current_adj.dim() != 3:
            raise ValueError(f"Expected current adjacency [B, N, N], got {tuple(current_adj.shape)}")

        batch_size, num_horizons, num_nodes, _ = z_graph.shape
        if current_adj.shape[0] != batch_size or current_adj.shape[1:] != (num_nodes, num_nodes):
            raise ValueError(
                f"Current adjacency shape mismatch: expected [B, {num_nodes}, {num_nodes}], got {tuple(current_adj.shape)}"
            )

        out = []
        for horizon_idx in range(num_horizons):
            z_h = z_graph[:, horizon_idx, :, :]
            out.append(self._dense_residual_horizon(z_h, current_adj, mlp, horizon_idx, num_horizons))
        return torch.stack(out, dim=1)

    def _predict_sparse_graphs(self, z_graph: torch.Tensor, current_adj: torch.Tensor, mlp: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        if z_graph.dim() != 4:
            raise ValueError(f"Expected z_graph [B, H, N, C], got {tuple(z_graph.shape)}")

        batch_size, num_horizons, num_nodes, _ = z_graph.shape
        edge_index_blocks = []
        edge_weight_blocks = []

        for horizon_idx in range(num_horizons):
            z_h = z_graph[:, horizon_idx, :, :]
            edge_index, edge_weight = self._sparse_residual_horizon(z_h, current_adj, mlp, horizon_idx, num_horizons)
            if edge_index.numel() > 0:
                edge_index_blocks.append(edge_index)
                edge_weight_blocks.append(edge_weight)

        if not edge_index_blocks:
            device = current_adj.device
            dtype = current_adj.dtype
            return (
                torch.empty((2, 0), device=device, dtype=torch.long),
                torch.empty((0,), device=device, dtype=dtype),
            )

        edge_index = torch.cat(edge_index_blocks, dim=1)
        edge_weight = torch.cat(edge_weight_blocks, dim=0)
        return edge_index, edge_weight

    def forward(
        self,
        z_graph: torch.Tensor,
        a_wind_current: torch.Tensor,
        a_sem_current: torch.Tensor,
        return_sparse: bool = False,
        candidate_scale: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Predict residual-adjusted future wind and semantic graphs, optionally as sparse edge lists."""
        if z_graph.dim() != 4:
            raise ValueError(f"Expected z_graph [B, H, N, C], got {tuple(z_graph.shape)}")
        if candidate_scale is not None:
            self.candidate_scale = max(1, int(candidate_scale))

        if return_sparse:
            a_wind_sparse = self._predict_sparse_graphs(z_graph, a_wind_current, self.wind_mlp)
            a_sem_sparse = self._predict_sparse_graphs(z_graph, a_sem_current, self.sem_mlp)
            return a_wind_sparse, a_sem_sparse

        a_wind_hat = self._predict_residuals(z_graph, a_wind_current, self.wind_mlp)
        a_sem_hat = self._predict_residuals(z_graph, a_sem_current, self.sem_mlp)
        return a_wind_hat, a_sem_hat