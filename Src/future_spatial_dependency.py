"""Utilities for generating future spatial dependency graphs (FSDG).

This module applies the forecast graph generation path used by the pipeline:
Z_hist [B, W, N, C] -> horizon-aware latent states [B, H, N, C] -> FSDG ->
Future Graphs -> FSDP.
"""

import torch
import torch.nn as nn


class FutureSpatialDependencyGenerator(nn.Module):
    """Generate future wind and semantic adjacency graphs from horizon latent node states.

    The generator scores only a horizon-specific candidate pool of likely neighbors,
    estimates a residual update for those candidates from the latent node states, and
    retains the strongest Top-K updates per destination node. Directed adjacency rows
    represent receivers (A_in[dst, src]); sparse edge lists remain [src, dst]. The
    resulting graphs support dense and sparse downstream propagation.
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
            latent_dim: Feature size of each node embedding.
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

        For each batch item and destination node, the method ranks candidate sources by
        incoming adjacency strength, removes self-loops, and returns likely senders for
        residual scoring at that forecast horizon.
        """
        if current_adj.dim() != 3:
            raise ValueError(f"Expected current adjacency [B, N, N], got {tuple(current_adj.shape)}")

        batch_size, num_nodes, _ = current_adj.shape
        candidate_k = self._candidate_size_for_horizon(num_horizons, horizon_idx)
        candidate_k = min(candidate_k, max(1, num_nodes - 1))

        candidate_pool = []
        for batch_idx in range(batch_size):
            row_pool = []
            for dst_idx in range(num_nodes):
                row = current_adj[batch_idx, dst_idx].clone()
                row[dst_idx] = -torch.inf
                if row.numel() == 0:
                    row_pool.append(torch.empty((0,), device=current_adj.device, dtype=torch.long))
                    continue
                if row.abs().sum() == 0:
                    src_idx = torch.arange(num_nodes, device=current_adj.device)
                    src_idx = src_idx[src_idx != dst_idx]
                    if src_idx.numel() == 0:
                        row_pool.append(torch.empty((0,), device=current_adj.device, dtype=torch.long))
                    else:
                        row_pool.append(src_idx[:candidate_k])
                    continue
                _, src_idx = torch.topk(row, k=min(candidate_k, row.numel()), largest=True, sorted=True)
                row_pool.append(src_idx)
            candidate_pool.append(torch.stack(row_pool, dim=0))
        return torch.stack(candidate_pool, dim=0)

    def _rowwise_residual_update(
        self,
        z_h: torch.Tensor,
        current_adj: torch.Tensor,
        mlp: nn.Module,
        horizon_idx: int,
        num_horizons: int,
        symmetric: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Apply candidate scoring and row-wise incoming-edge Top-K pruning.

        Returns the fully updated receiver-row adjacency and selected edge indices in
        conventional source/destination order. Both output builders use this helper so
        dense and sparse representations stay consistent.
        """
        batch_size, num_nodes, _ = z_h.shape
        candidate_pool = self._candidate_pool_for_horizon(current_adj, horizon_idx, num_horizons)
        out_adj = current_adj.clone()
        selected_rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        if num_nodes <= 0:
            return out_adj, selected_rows

        batch_idx = torch.arange(batch_size, device=current_adj.device).view(batch_size, 1, 1)
        dst_idx = torch.arange(num_nodes, device=current_adj.device).view(1, num_nodes, 1)
        candidate_k = candidate_pool.size(-1)

        if candidate_k <= 0:
            return out_adj, selected_rows

        dst_state = z_h[:, :, None, :].expand(-1, -1, candidate_k, -1)
        src_state = z_h[batch_idx, candidate_pool, :]
        edge_features = torch.cat([src_state, dst_state], dim=-1)
        flat_edge_features = edge_features.reshape(-1, edge_features.size(-1))

        residuals = mlp(flat_edge_features).reshape(batch_size, num_nodes, candidate_k).squeeze(-1) * self.residual_scale
        current_vals = current_adj.gather(2, candidate_pool).clamp(min=0.0, max=1.0)

        valid_mask = candidate_pool.ne(dst_idx.expand_as(candidate_pool))
        candidate_updates = torch.clamp(current_vals + residuals, min=0.0, max=1.0)
        candidate_updates = torch.where(valid_mask, candidate_updates, torch.zeros_like(candidate_updates))
        out_adj.scatter_(2, candidate_pool, candidate_updates)

        for batch_idx in range(batch_size):
            for dst_idx in range(num_nodes):
                row = out_adj[batch_idx, dst_idx].clone()
                row[dst_idx] = -torch.inf
                top_k = min(self.k, max(0, num_nodes - 1))
                if top_k <= 0:
                    selected_rows.append((
                        torch.empty((0,), device=current_adj.device, dtype=torch.long),
                        torch.empty((0,), device=current_adj.device, dtype=torch.long),
                        torch.empty((0,), device=current_adj.device, dtype=current_adj.dtype),
                    ))
                    continue
                _, top_pos = torch.topk(row, k=top_k, largest=True, sorted=True)
                out_adj[batch_idx, dst_idx] = 0.0
                out_adj[batch_idx, dst_idx, top_pos] = row[top_pos]

                if symmetric:
                    continue

                src_nodes = top_pos + batch_idx * num_nodes
                dst_nodes = torch.full((top_pos.numel(),), dst_idx, device=current_adj.device, dtype=torch.long) + batch_idx * num_nodes
                selected_rows.append((src_nodes, dst_nodes, row[top_pos]))

        if symmetric:
            out_adj = torch.maximum(out_adj, out_adj.transpose(-1, -2))
            selected_rows = []
            for batch_idx in range(batch_size):
                for dst_idx in range(num_nodes):
                    row = out_adj[batch_idx, dst_idx].clone()
                    row[dst_idx] = 0.0
                    pos = torch.nonzero(row, as_tuple=False).flatten()
                    if pos.numel() == 0:
                        selected_rows.append((
                            torch.empty((0,), device=current_adj.device, dtype=torch.long),
                            torch.empty((0,), device=current_adj.device, dtype=torch.long),
                            torch.empty((0,), device=current_adj.device, dtype=current_adj.dtype),
                        ))
                        continue
                    src_nodes = pos + batch_idx * num_nodes
                    dst_nodes = torch.full((pos.numel(),), dst_idx, device=current_adj.device, dtype=torch.long) + batch_idx * num_nodes
                    selected_rows.append((src_nodes, dst_nodes, row[pos]))

        return out_adj, selected_rows

    def _dense_residual_horizon(
        self,
        z_h: torch.Tensor,
        current_adj: torch.Tensor,
        mlp: nn.Module,
        horizon_idx: int,
        num_horizons: int,
        symmetric: bool = False,
    ) -> torch.Tensor:
        """Build one dense future adjacency matrix for a single horizon.

        Each destination row is narrowed to likely source nodes, scored, and pruned to its
        strongest incoming Top-K edges. The result uses ``A_in[dst, src]`` layout.
        """
        out_adj, _ = self._rowwise_residual_update(z_h, current_adj, mlp, horizon_idx, num_horizons, symmetric=symmetric)
        return out_adj

    def _sparse_residual_horizon(
        self,
        z_h: torch.Tensor,
        current_adj: torch.Tensor,
        mlp: nn.Module,
        horizon_idx: int,
        num_horizons: int,
        symmetric: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the sparse edge list for one horizon after final row-wise Top-K pruning.

        Each destination evaluates likely incoming source nodes and prunes the updated row to
        Top-K. The returned edge indices use conventional ``[src, dst]`` ordering and exactly
        represent the corresponding dense receiver-row graph.
        """
        _, selected_rows = self._rowwise_residual_update(z_h, current_adj, mlp, horizon_idx, num_horizons, symmetric=symmetric)

        if symmetric:
            edge_index_blocks = []
            edge_weight_blocks = []
            for src_nodes, dst_nodes, selected_vals in selected_rows:
                if src_nodes.numel() == 0:
                    continue
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

        edge_index_blocks = []
        edge_weight_blocks = []

        for src_nodes, dst_nodes, selected_vals in selected_rows:
            if src_nodes.numel() == 0:
                continue
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

    def _predict_dense_graphs(
        self,
        z_graph: torch.Tensor,
        current_adj: torch.Tensor,
        mlp: nn.Module,
        symmetric: bool = False,
    ) -> torch.Tensor:
        """Predict dense future adjacency matrices for every horizon in a batch.

        The method evaluates a candidate-filtered residual update for each horizon, then
        reconstructs a dense [B, H, N, N] graph by updating the candidate entries in the full
        adjacency row and applying final Top-K selection over that row. Unselected edges are
        zeroed to reflect the final sparsified future graph.

        Args:
            z_graph: Latent node embeddings of shape [B, H, N, C].
            current_adj: Dense incoming adjacency matrix [B, N, N], indexed [dst, src].
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
            out.append(self._dense_residual_horizon(z_h, current_adj, mlp, horizon_idx, num_horizons, symmetric=symmetric))
        return torch.stack(out, dim=1)

    def _predict_sparse_graphs(
        self,
        z_graph: torch.Tensor,
        current_adj: torch.Tensor,
        mlp: nn.Module,
        symmetric: bool = False,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Predict a per-horizon sparse edge-list representation.

        Each horizon independently scores only its candidate neighbors, keeps the strongest
        Top-K incoming edges per destination node, and returns standard [src, dst] edge
        indices in a structured list indexed by horizon.
        """
        if z_graph.dim() != 4:
            raise ValueError(f"Expected z_graph [B, H, N, C], got {tuple(z_graph.shape)}")

        _, num_horizons, _, _ = z_graph.shape
        horizon_graphs = []

        for horizon_idx in range(num_horizons):
            z_h = z_graph[:, horizon_idx, :, :]
            edge_index, edge_weight = self._sparse_residual_horizon(z_h, current_adj, mlp, horizon_idx, num_horizons, symmetric=symmetric)
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
        most likely incoming neighbors, scores residual updates only on those candidates, and
        keeps the strongest Top-K edges per destination. Dense matrices use [dst, src] layout;
        sparse edge indices use [src, dst].

        Args:
            z_graph: Latent future states of shape [B, H, N, C].
            a_wind_current_dense: Dense incoming wind adjacency [B, N, N], indexed [dst, src].
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
            a_wind_sparse = self._predict_sparse_graphs(z_graph, a_wind_current_dense, self.wind_mlp, symmetric=False)
            a_sem_sparse = self._predict_sparse_graphs(z_graph, a_sem_current_dense, self.sem_mlp, symmetric=True)
            return a_wind_sparse, a_sem_sparse

        a_wind_hat = self._predict_dense_graphs(z_graph, a_wind_current_dense, self.wind_mlp, symmetric=False)
        a_sem_hat = self._predict_dense_graphs(z_graph, a_sem_current_dense, self.sem_mlp, symmetric=True)
        return a_wind_hat, a_sem_hat