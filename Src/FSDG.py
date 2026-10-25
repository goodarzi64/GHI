"""Utilities for generating future spatial dependency graphs (FSDG).

This module applies the forecast graph generation path used by the pipeline:
Z_LGEE [B, W, N, C] -> horizon-aware latent states [B, H, N, C] -> FSDG ->
Future Graphs -> FSDP.
"""

import torch
import torch.nn as nn


class FutureSpatialDependencyGenerator(nn.Module):
    """Generate future wind and semantic adjacency graphs from horizon latent node states.

    The generator scores every node pair and applies a residual update to the dense current
    adjacency without edge pruning. Directed rows represent receivers (A_in[dst, src]).
    Future graphs are returned as dense tensors for regularization and propagation.
    With zero output initialization, the initial output is the clipped current graph with its
    diagonal zeroed; the semantic graph is symmetrized.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int | None = None,
        residual_scale: float = 0.1,
        zero_init_output: bool = True,
    ):
        """Initialize the dense graph predictor for wind and semantic dependencies.

        Args:
            latent_dim: Feature size of each node embedding.
            hidden_dim: Hidden width of the edge-scoring MLPs. Defaults to 2 * latent_dim.
            residual_scale: Scaling factor applied to learned residual updates on every pair.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.residual_scale = residual_scale
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
        if zero_init_output:
            for mlp in (self.wind_mlp, self.sem_mlp):
                nn.init.zeros_(mlp[-1].weight)
                nn.init.zeros_(mlp[-1].bias)

    def _rowwise_residual_update(
        self,
        z_h: torch.Tensor,
        current_adj: torch.Tensor,
        mlp: nn.Module,
        symmetric: bool = False,
    ) -> torch.Tensor:
        """Score all node pairs and apply residual updates without edge pruning."""
        batch_size, num_nodes, _ = z_h.shape

        if num_nodes <= 0:
            return current_adj.clone()

        # Dense receiver-row layout: row is destination, column is source. The MLP
        # receives pair features in [z_dst, z_src] order.
        src_state = z_h[:, None, :, :].expand(-1, num_nodes, -1, -1)
        dst_state = z_h[:, :, None, :].expand(-1, -1, num_nodes, -1)
        edge_features = torch.cat([dst_state, src_state], dim=-1)
        residuals = mlp(edge_features.reshape(-1, edge_features.size(-1)))
        residuals = residuals.reshape(batch_size, num_nodes, num_nodes) * self.residual_scale
        out_adj = torch.clamp(current_adj + residuals, min=0.0, max=1.0)
        diagonal = torch.eye(num_nodes, device=current_adj.device, dtype=torch.bool).unsqueeze(0)
        out_adj = out_adj.masked_fill(diagonal, 0.0)

        if symmetric:
            out_adj = torch.maximum(out_adj, out_adj.transpose(-1, -2))

        return out_adj

    def _predict_dense_graphs(
        self,
        z_graph: torch.Tensor,
        current_adj: torch.Tensor,
        mlp: nn.Module,
        symmetric: bool = False,
    ) -> torch.Tensor:
        """Predict dense future adjacency matrices for every horizon in a batch.

        The method evaluates a dense residual update for every pair at each horizon. All
        off-diagonal values are retained for downstream sparsity regularization.

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
            out_adj = self._rowwise_residual_update(z_h, current_adj, mlp, symmetric=symmetric)
            out.append(out_adj)
        return torch.stack(out, dim=1)

    def forward(
        self,
        z_graph: torch.Tensor,
        a_wind_current_dense: torch.Tensor,
        a_sem_current_dense: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict future wind and semantic graphs from dense current adjacency matrices.

        For each horizon, the model scores all node pairs and updates the dense current graph
        without pruning. Dense matrices use [dst, src] layout.

        Args:
            z_graph: Latent future states of shape [B, H, N, C].
            a_wind_current_dense: Dense incoming wind adjacency [B, N, N], indexed [dst, src].
            a_sem_current_dense: Dense current semantic adjacency matrix [B, N, N].
        Returns:
            Dense future adjacency tensors of shape [B, H, N, N] for both graph branches.
        """
        if z_graph.dim() != 4:
            raise ValueError(f"Expected z_graph [B, H, N, C], got {tuple(z_graph.shape)}")

        a_wind_hat = self._predict_dense_graphs(z_graph, a_wind_current_dense, self.wind_mlp, symmetric=False)
        a_sem_hat = self._predict_dense_graphs(z_graph, a_sem_current_dense, self.sem_mlp, symmetric=True)
        return a_wind_hat, a_sem_hat
