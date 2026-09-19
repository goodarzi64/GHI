import torch
import torch.nn as nn


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


class CurrentStateRefinement(nn.Module):
    """Refine the current latent state before future graph propagation."""

    def __init__(self, channels: int, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        hidden_dim = 4 * channels if hidden_dim is None else hidden_dim

        self.norm = nn.LayerNorm(channels)
        self.proj_in = nn.Linear(channels, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj_out = nn.Linear(hidden_dim, channels)

    def forward(self, z_current: torch.Tensor) -> torch.Tensor:
        """Refine [B, N, C] into a propagation-ready state [B, N, C]."""
        if z_current.dim() != 3:
            raise ValueError(f"Expected current state [B, N, C], got {tuple(z_current.shape)}")

        residual = z_current
        x = self.norm(z_current)
        x = self.proj_in(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.proj_out(x)
        return residual + x


class HorizonAwarePropagationGate(nn.Module):
    """Learn node-wise, feature-wise gates for fusing multi-graph messages."""

    def __init__(self, channels: int, horizon_dim: int, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.horizon_dim = horizon_dim
        self.net = nn.Sequential(
            nn.Linear(channels + horizon_dim, 2 * channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * channels, 3 * channels),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor, horizon_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return gates [g_phys, g_wind, g_sem] with shape [B, N, C]."""
        if h.dim() != 3:
            raise ValueError(f"Expected hidden state [B, N, C], got {tuple(h.shape)}")

        h_in = torch.cat([h, horizon_emb.unsqueeze(1).expand(-1, h.shape[1], -1)], dim=-1)
        gates = self.net(h_in)
        g_phys, g_wind, g_sem = torch.chunk(gates, 3, dim=-1)
        return g_phys, g_wind, g_sem


class MultiGraphAdaptivePropagation(nn.Module):
    """Propagate a single shared hidden state over future graph views for each horizon."""

    def __init__(self, channels: int, num_horizons: int, propagation_steps: int = 3, alpha: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.num_horizons = num_horizons
        self.propagation_steps = propagation_steps
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)

        self.horizon_embeddings = nn.Parameter(torch.randn(num_horizons, channels))
        self.gate = HorizonAwarePropagationGate(channels, channels)

    def _normalise_adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        if adj.dim() != 3:
            raise ValueError(f"Expected adjacency [B, N, N], got {tuple(adj.shape)}")
        denom = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return adj / denom

    def forward(
        self,
        z_refined: torch.Tensor,
        a_phys: torch.Tensor,
        a_wind_hat: torch.Tensor,
        a_sem_hat: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate [B, N, C] state over [B, H, N, N] future graphs.

        Returns:
            [B, H, N, C]
        """
        if z_refined.dim() != 3:
            raise ValueError(f"Expected refined state [B, N, C], got {tuple(z_refined.shape)}")
        if a_wind_hat.dim() != 4 or a_sem_hat.dim() != 4:
            raise ValueError(f"Expected future graph tensors [B, H, N, N], got {tuple(a_wind_hat.shape)} and {tuple(a_sem_hat.shape)}")

        batch_size, num_nodes, channels = z_refined.shape
        horizon_emb = self.horizon_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        a_phys_norm = self._normalise_adjacency(a_phys.unsqueeze(0).expand(batch_size, -1, -1))

        outputs = []
        for horizon_idx in range(self.num_horizons):
            h = z_refined
            h0 = z_refined
            a_wind_h = self._normalise_adjacency(a_wind_hat[:, horizon_idx, :, :])
            a_sem_h = self._normalise_adjacency(a_sem_hat[:, horizon_idx, :, :])
            for _ in range(self.propagation_steps):
                g_phys, g_wind, g_sem = self.gate(h, horizon_emb[:, horizon_idx, :])

                m_phys = torch.einsum('bni,bnc->bic', a_phys_norm, h)
                m_wind = torch.einsum('bni,bnc->bic', a_wind_h, h)
                m_sem = torch.einsum('bni,bnc->bic', a_sem_h, h)

                m = g_phys * m_phys + g_wind * m_wind + g_sem * m_sem
                h = (1.0 - self.alpha) * m + self.alpha * h0
                h = self.dropout(h)

            outputs.append(h)

        return torch.stack(outputs, dim=1)