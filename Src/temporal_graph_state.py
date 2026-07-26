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
    def __init__(self, state_dim: int, hidden_dim: int = 32, n_horizons: int = 1, horizon_emb_dim: int | None = None):
        super().__init__()
        self.state_dim = state_dim
        self.n_horizons = n_horizons
        self.horizon_emb_dim = state_dim if horizon_emb_dim is None else horizon_emb_dim

        # MLP input size: current_edge(1) + src + dst + diff + hadamard (4*state_dim) + horizon_emb
        feat_in = 1 + 4 * state_dim + self.horizon_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(feat_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # learnable embeddings for each horizon index
        if self.n_horizons > 1:
            self.horizon_embeddings = nn.Embedding(self.n_horizons, self.horizon_emb_dim)
        else:
            # single horizon: use a learnable vector of shape [1, emb_dim]
            self.horizon_embeddings = nn.Parameter(torch.zeros(1, self.horizon_emb_dim))

    def forward(self, historical_adj: torch.Tensor, node_state: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return predicted edge weights for each horizon.

        Args:
            historical_adj: [B, N, N]
            node_state: [B, N, C]
            edge_index: [2, E]

        Returns:
            predictions: [B, H, E]
        """
        if historical_adj.dim() != 3:
            raise ValueError(f"Expected historical adjacency [B, N, N], got {tuple(historical_adj.shape)}")
        if node_state.dim() != 3:
            raise ValueError(f"Expected node_state [B, N, C], got {tuple(node_state.shape)}")

        B = historical_adj.shape[0]
        E = edge_index.shape[1]

        src_state = node_state[:, edge_index[0], :]  # [B, E, C]
        dst_state = node_state[:, edge_index[1], :]  # [B, E, C]
        current_edge = historical_adj[:, edge_index[0], edge_index[1]].unsqueeze(-1)  # [B, E, 1]

        diff_state = torch.abs(src_state - dst_state)  # [B, E, C]
        hadamard_state = src_state * dst_state  # [B, E, C]

        # Prepare horizon embeddings: [H, emb_dim]
        if isinstance(self.horizon_embeddings, nn.Parameter):
            horizon_embs = self.horizon_embeddings.unsqueeze(0)  # [1, emb_dim]
        else:
            horizon_embs = self.horizon_embeddings.weight  # [H, emb_dim]

        H = horizon_embs.shape[0]

        # Expand tensors to [B, H, E, ...]
        src_exp = src_state.unsqueeze(1).expand(B, H, E, self.state_dim)
        dst_exp = dst_state.unsqueeze(1).expand(B, H, E, self.state_dim)
        diff_exp = diff_state.unsqueeze(1).expand(B, H, E, self.state_dim)
        had_exp = hadamard_state.unsqueeze(1).expand(B, H, E, self.state_dim)
        edge_exp = current_edge.unsqueeze(1).expand(B, H, E, 1)

        # horizon embeddings -> [1, H, 1, emb_dim] -> expand to [B, H, E, emb_dim]
        h_emb = horizon_embs.view(1, H, 1, -1).expand(B, H, E, -1)

        feat = torch.cat([edge_exp, src_exp, dst_exp, diff_exp, had_exp, h_emb], dim=-1)
        Bn, Hn, En, F = feat.shape

        feat_flat = feat.reshape(Bn * Hn * En, F)
        delta_flat = self.mlp(feat_flat).squeeze(-1)
        delta = delta_flat.view(Bn, Hn, En)

        current_edge_flat = current_edge.unsqueeze(1).expand(B, H, E)
        return current_edge_flat + delta


class HorizonAwareMultiGraphAPPNP(nn.Module):
    """Decoder module for shared multi-graph adaptive propagation.

    Implements one hidden state H(k) propagated over three graph views
    (physical, wind, semantic) with horizon-conditioned feature-wise gates.
    The propagation is repeated for a fixed number of steps and returns the
    final horizon-specific propagated embedding.
    """

    def __init__(
        self,
        channels: int,
        horizon_emb_dim: int | None = None,
        propagation_steps: int = 3,
        alpha: float = 0.1,
        dropout: float = 0.1,
        num_horizons: int | None = None,
    ):
        super().__init__()
        self.channels = channels
        self.horizon_emb_dim = channels if horizon_emb_dim is None else horizon_emb_dim
        self.propagation_steps = propagation_steps
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)

        if num_horizons is not None and num_horizons > 1:
            self.horizon_embeddings = nn.Embedding(num_horizons, self.horizon_emb_dim)
        else:
            self.horizon_embeddings = nn.Parameter(torch.zeros(1, self.horizon_emb_dim))

        gate_hidden_dim = max(8, self.channels)
        self.gate_net = nn.Sequential(
            nn.Linear(self.channels + self.horizon_emb_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 3 * self.channels),
            nn.Sigmoid(),
        )

    def _normalise_adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        if adj.dim() != 3:
            raise ValueError(f"Expected adjacency [B, N, N], got {tuple(adj.shape)}")
        if adj.shape[-1] != adj.shape[-2]:
            raise ValueError(f"Expected square adjacency matrix, got {tuple(adj.shape)}")

        denom = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return adj / denom

    def _get_horizon_embedding(self, batch_size: int, device: torch.device, horizon_idx: int | torch.Tensor | None) -> torch.Tensor:
        if isinstance(self.horizon_embeddings, nn.Parameter):
            return self.horizon_embeddings.expand(batch_size, -1).to(device)

        if horizon_idx is None:
            horizon_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        elif isinstance(horizon_idx, int):
            horizon_idx = torch.full((batch_size,), horizon_idx, dtype=torch.long, device=device)
        else:
            horizon_idx = horizon_idx.to(device)
            if horizon_idx.dim() == 0:
                horizon_idx = horizon_idx.unsqueeze(0).expand(batch_size)
            elif horizon_idx.shape[0] != batch_size:
                horizon_idx = horizon_idx[:batch_size]

        return self.horizon_embeddings(horizon_idx)

    def forward(
        self,
        z: torch.Tensor,
        adj_phys: torch.Tensor,
        adj_wind: torch.Tensor,
        adj_sem: torch.Tensor,
        horizon_idx: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Propagate a shared hidden state over the three graph views.

        Args:
            z: [B, N, C] initial hidden state
            adj_phys: [B, N, N] physical adjacency
            adj_wind: [B, N, N] wind adjacency
            adj_sem: [B, N, N] semantic adjacency
            horizon_idx: optional horizon index for embedding lookup

        Returns:
            [B, N, C] propagated embedding for the selected horizon
        """
        if z.dim() != 3:
            raise ValueError(f"Expected node state [B, N, C], got {tuple(z.shape)}")

        batch_size, num_nodes, _ = z.shape
        h = z
        h0 = z
        horizon_emb = self._get_horizon_embedding(batch_size, z.device, horizon_idx)

        for _ in range(self.propagation_steps):
            h_in = torch.cat([h, horizon_emb.unsqueeze(1).expand(-1, num_nodes, -1)], dim=-1)
            gates = self.gate_net(h_in)
            g_phys, g_wind, g_sem = torch.chunk(gates, 3, dim=-1)

            adj_phys_norm = self._normalise_adjacency(adj_phys)
            adj_wind_norm = self._normalise_adjacency(adj_wind)
            adj_sem_norm = self._normalise_adjacency(adj_sem)

            m_phys = torch.einsum('bni,bnc->bic', adj_phys_norm, h)
            m_wind = torch.einsum('bni,bnc->bic', adj_wind_norm, h)
            m_sem = torch.einsum('bni,bnc->bic', adj_sem_norm, h)

            m = g_phys * m_phys + g_wind * m_wind + g_sem * m_sem
            h = (1.0 - self.alpha) * m + self.alpha * h0
            h = self.dropout(h)

        return h


class ForecastHead(nn.Module):
    """Simple linear forecast head for horizon-specific GHI prediction."""

    def __init__(self, channels: int, num_horizons: int = 1, horizon_emb_dim: int | None = None):
        super().__init__()
        self.channels = channels
        self.num_horizons = num_horizons
        self.horizon_emb_dim = channels if horizon_emb_dim is None else horizon_emb_dim

        if self.num_horizons > 1:
            self.horizon_embeddings = nn.Embedding(self.num_horizons, self.horizon_emb_dim)
        else:
            self.horizon_embeddings = nn.Parameter(torch.zeros(1, self.horizon_emb_dim))

        self.proj = nn.Linear(channels + self.horizon_emb_dim, 1)

    def _get_horizon_embedding(self, batch_size: int, device: torch.device, horizon_idx: int | torch.Tensor | None) -> torch.Tensor:
        if isinstance(self.horizon_embeddings, nn.Parameter):
            return self.horizon_embeddings.expand(batch_size, -1).to(device)

        if horizon_idx is None:
            horizon_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        elif isinstance(horizon_idx, int):
            horizon_idx = torch.full((batch_size,), horizon_idx, dtype=torch.long, device=device)
        else:
            horizon_idx = horizon_idx.to(device)
            if horizon_idx.dim() == 0:
                horizon_idx = horizon_idx.unsqueeze(0).expand(batch_size)
            elif horizon_idx.shape[0] != batch_size:
                horizon_idx = horizon_idx[:batch_size]

        return self.horizon_embeddings(horizon_idx)

    def forward(self, state: torch.Tensor, horizon_idx: int | torch.Tensor | None = None) -> torch.Tensor:
        """Map propagated state [B, N, C] to predictions [B, H, N]."""
        if state.dim() != 3:
            raise ValueError(f"Expected state [B, N, C], got {tuple(state.shape)}")

        batch_size, num_nodes, _ = state.shape
        horizon_emb = self._get_horizon_embedding(batch_size, state.device, horizon_idx)
        horizon_emb = horizon_emb.unsqueeze(1).expand(-1, num_nodes, -1)

        feats = torch.cat([state, horizon_emb], dim=-1)
        logits = self.proj(feats).squeeze(-1)  # [B, N]
        if isinstance(horizon_idx, torch.Tensor) and horizon_idx.dim() > 0 and horizon_idx.shape[0] == batch_size:
            return logits.unsqueeze(1).expand(-1, self.num_horizons, -1)

        return logits.unsqueeze(1).expand(-1, self.num_horizons, -1)


class ForecastAndGraphLoss(nn.Module):
    """Composite forecast and graph reconstruction loss."""

    def __init__(
        self,
        forecast_weight: float = 1.0,
        graph_weight: float = 0.5,
        sparsity_weight: float = 0.0,
        delta: float = 1.0,
    ):
        super().__init__()
        self.forecast_weight = forecast_weight
        self.graph_weight = graph_weight
        self.sparsity_weight = sparsity_weight
        self.delta = delta

    def _huber_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        abs_diff = diff.abs()
        quad = 0.5 * diff.pow(2)
        lin = self.delta * (abs_diff - 0.5 * self.delta)
        return torch.where(abs_diff <= self.delta, quad, lin).mean()

    def _graph_reconstruction_loss(self, pred_graph: torch.Tensor, target_graph: torch.Tensor) -> torch.Tensor:
        if pred_graph.dim() != target_graph.dim():
            raise ValueError(f"Expected matching graph tensor dims, got {tuple(pred_graph.shape)} and {tuple(target_graph.shape)}")
        return self._huber_loss(pred_graph, target_graph)

    def _sparsity_penalty(self, graph: torch.Tensor) -> torch.Tensor:
        return graph.abs().mean()

    def forward(
        self,
        forecast_pred: torch.Tensor,
        forecast_target: torch.Tensor,
        wind_graph_pred: torch.Tensor,
        wind_graph_target: torch.Tensor,
        sem_graph_pred: torch.Tensor,
        sem_graph_target: torch.Tensor,
    ) -> torch.Tensor:
        forecast_loss = self._huber_loss(forecast_pred, forecast_target)
        graph_loss = (
            self._graph_reconstruction_loss(wind_graph_pred, wind_graph_target)
            + self._graph_reconstruction_loss(sem_graph_pred, sem_graph_target)
        ) / 2.0

        sparsity_penalty = 0.0
        if self.sparsity_weight > 0:
            sparsity_penalty = self.sparsity_weight * (
                self._sparsity_penalty(wind_graph_pred) + self._sparsity_penalty(sem_graph_pred)
            ) / 2.0

        return self.forecast_weight * forecast_loss + self.graph_weight * graph_loss + sparsity_penalty
