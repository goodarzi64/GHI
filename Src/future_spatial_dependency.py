"""Utilities for generating future spatial dependency graphs(FSDG).

This module applies the remaining forecast graph generation path used by the
pipeline: Z_hist [B, W, N, C] -> LGEE -> Z_graph [B, H, N, C] -> FSDG ->
Future Graphs -> FSDP.
"""

import torch
import torch.nn as nn


class FutureSpatialDependencyGenerator(nn.Module):
    """Generate future wind and semantic adjacency graphs from LGEE node states.

    The generator scores only a horizon-specific candidate pool of likely neighbors,
    estimates a residual update for those candidates from the latent node states, and
    retains the strongest Top-K updates per source node. The resulting graphs support
    both dense matrix output and sparse edge-list output for downstream propagation.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int | None = None,
        residual_scale: float = 0.1,
        k: int = 5,
        candidate_scale: int = 3,
    ):
        """Initialize the candidate-filtered graph predictor for wind and semantic dependencies.

        Args:
            latent_dim: Feature size of each node embedding from LGEE.
            hidden_dim: Hidden width of the edge-scoring MLPs. Defaults to 2 * latent_dim.
            residual_scale: Scaling factor applied to learned residual updates on the selected
                candidate edges.
            k: Number of final edges retained per node after Top-K selection.
            candidate_scale: Multiplicative expansion factor for the candidate pool in far
                horizons.
        """
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
            nn.Linear(hidden_dim, 1),
        )
        self.sem_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, 1),
        )

    def _candidate_size_for_horizon(self, num_horizons: int, horizon_idx: int) -> int:
        """Return the candidate pool size for a specific future horizon.

        Early horizons keep a tighter candidate pool while farther horizons expand to allow
        broader structural evolution. This creates a depth-aware search space without
        evaluating every possible neighbor.
        """
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
        """Build a horizon-aware per-node candidate pool from the current dense adjacency.

        For each batch item and each source node, the method ranks candidate destinations by
        the current adjacency strength, removes self-loops, and returns the most relevant
        neighbors for residual scoring at that forecast horizon.
        """
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

    def _dense_residual_horizon(self, z_h: torch.Tensor, current_adj: torch.Tensor, mlp: nn.Module, horizon_idx: int, num_horizons: int) -> torch.Tensor:
        """Build one dense future adjacency matrix for a single horizon.

        For each source node, the method restricts scoring to a horizon-aware candidate pool,
        estimates residual updates only for those likely neighbors, and keeps the strongest
        Top-K updates in the returned dense matrix. Unselected entries remain zero.
        """
        batch_size, num_nodes, _ = z_h.shape
        candidate_pool = self._candidate_pool_for_horizon(current_adj, horizon_idx, num_horizons)
        out_adj = torch.zeros_like(current_adj)

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
                residuals = mlp(edge_features).squeeze(-1) * self.residual_scale
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
        """Return the sparse edge list for one horizon using candidate-filtered Top-K residuals.

        Each source node evaluates only its horizon-aware candidate neighbors, scores the
        resulting residual updates, and keeps the strongest Top-K edges. The output is a
        compact edge list for this single horizon and can be concatenated across horizons.
        """
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
                residuals = mlp(edge_features).squeeze(-1) * self.residual_scale
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

    def _predict_dense_graphs(self, z_graph: torch.Tensor, current_adj: torch.Tensor, mlp: nn.Module) -> torch.Tensor:
        """Predict dense future adjacency matrices for every horizon in a batch.

        The method evaluates a candidate-filtered residual update for each horizon, then
        reconstructs a dense [B, H, N, N] graph by retaining only the strongest Top-K edges
        per source node. Entries not selected for the Top-K set remain zero.

        Args:
            z_graph: Latent node embeddings of shape [B, H, N, C].
            current_adj: Dense current adjacency matrix of shape [B, N, N].
            mlp: Residual scoring network for the relevant graph branch.

        Returns:
            Dense future adjacency matrices of shape [B, H, N, N].
        """
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

    def _predict_sparse_graphs(self, z_graph: torch.Tensor, current_adj: torch.Tensor, mlp: nn.Module) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Predict a per-horizon sparse edge-list representation.

        Each horizon independently scores only its candidate neighbors, keeps the strongest
        Top-K edges per source node, and returns the result as a structured list indexed by
        horizon. This preserves horizon identity without flattening all future graphs together.
        """
        if z_graph.dim() != 4:
            raise ValueError(f"Expected z_graph [B, H, N, C], got {tuple(z_graph.shape)}")

        _, num_horizons, _, _ = z_graph.shape
        horizon_graphs = []

        for horizon_idx in range(num_horizons):
            z_h = z_graph[:, horizon_idx, :, :]
            edge_index, edge_weight = self._sparse_residual_horizon(z_h, current_adj, mlp, horizon_idx, num_horizons)
            horizon_graphs.append((edge_index, edge_weight))

        return horizon_graphs

    def forward(
        self,
        z_graph: torch.Tensor,
        a_wind_current_dense: torch.Tensor,
        a_sem_current_dense: torch.Tensor,
        return_sparse: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]]:
        """Predict future wind and semantic graphs from dense current adjacency matrices.

        For each horizon, the model first narrows the search space to a candidate pool of the
        most likely neighbors, scores residual updates only on those candidates, and keeps the
        strongest Top-K edges per source node. Returned graphs may be dense or sparse depending
        on the chosen output format.

        Args:
            z_graph: Latent future states of shape [B, H, N, C].
            a_wind_current_dense: Dense current wind adjacency matrix [B, N, N].
            a_sem_current_dense: Dense current semantic adjacency matrix [B, N, N].
            return_sparse: If True, return a list of sparse edge-index/edge-weight pairs, one
                per horizon; otherwise return dense future adjacency matrices [B, H, N, N].

        Returns:
            Either dense future adjacency tensors or a per-horizon sparse representation for
            both graph branches.
        """
        if z_graph.dim() != 4:
            raise ValueError(f"Expected z_graph [B, H, N, C], got {tuple(z_graph.shape)}")

        if return_sparse:
            a_wind_sparse = self._predict_sparse_graphs(z_graph, a_wind_current_dense, self.wind_mlp)
            a_sem_sparse = self._predict_sparse_graphs(z_graph, a_sem_current_dense, self.sem_mlp)
            return a_wind_sparse, a_sem_sparse

        a_wind_hat = self._predict_dense_graphs(z_graph, a_wind_current_dense, self.wind_mlp)
        a_sem_hat = self._predict_dense_graphs(z_graph, a_sem_current_dense, self.sem_mlp)
        return a_wind_hat, a_sem_hat