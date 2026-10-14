import torch
import torch.nn as nn


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
            nn.Linear(4 * channels + horizon_dim, 2 * channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * channels, 3 * channels),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h: torch.Tensor,
        m_phys: torch.Tensor,
        m_wind: torch.Tensor,
        m_sem: torch.Tensor,
        horizon_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return gates [g_phys, g_wind, g_sem] with shape [B, N, C]."""
        if h.dim() != 3:
            raise ValueError(f"Expected hidden state [B, N, C], got {tuple(h.shape)}")

        h_in = torch.cat(
            [
                h,
                m_phys,
                m_wind,
                m_sem,
                horizon_emb.unsqueeze(1).expand(-1, h.shape[1], -1),
            ],
            dim=-1,
        )
        gates = self.net(h_in)
        g_phys, g_wind, g_sem = torch.chunk(gates, 3, dim=-1)
        return g_phys, g_wind, g_sem


class HorizonAwareMultiGraphAPPNP(nn.Module):
    """Backward-compatible decoder wrapper for the previous APPNP-style API.

    The current maintained implementation is ``MultiGraphAdaptivePropagation``. This
    compatibility shim preserves the older constructor and call signature used by the
    repository tests while delegating to the maintained propagation logic.
    """

    def __init__(
        self,
        channels: int,
        horizon_emb_dim: int = 4,
        propagation_steps: int = 3,
        alpha: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.horizon_emb_dim = horizon_emb_dim
        self.propagator = MultiGraphAdaptivePropagation(
            channels=channels,
            num_horizons=1,
            propagation_steps=propagation_steps,
            alpha=alpha,
            dropout=dropout,
            use_refinement=False,
        )

    def forward(
        self,
        z: torch.Tensor,
        adj_phys: torch.Tensor,
        adj_wind: torch.Tensor,
        adj_sem: torch.Tensor,
        horizon_idx: int | None = None,
    ) -> torch.Tensor:
        """Return a propagated state of shape [B, N, C]."""
        if z.dim() != 3:
            raise ValueError(f"Expected z [B, N, C], got {tuple(z.shape)}")

        batch_size, num_nodes, _ = z.shape

        if adj_phys.dim() == 3 and adj_phys.shape[0] == batch_size and adj_phys.shape[1:] == (num_nodes, num_nodes):
            adj_phys = adj_phys[0]
        elif adj_phys.dim() == 3 and adj_phys.shape[0] == 1 and adj_phys.shape[1:] == (num_nodes, num_nodes):
            adj_phys = adj_phys[0]
        elif adj_phys.dim() != 2:
            raise ValueError(f"Expected adj_phys [N, N] or [B, N, N], got {tuple(adj_phys.shape)}")

        if adj_wind.dim() == 3 and adj_wind.shape[1:] == (num_nodes, num_nodes):
            adj_wind = adj_wind.unsqueeze(1)
        elif adj_wind.dim() == 2 and adj_wind.shape == (num_nodes, num_nodes):
            adj_wind = adj_wind.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, num_nodes, num_nodes)
        else:
            raise ValueError(f"Expected adj_wind [B, N, N] or [N, N], got {tuple(adj_wind.shape)}")

        if adj_sem.dim() == 3 and adj_sem.shape[1:] == (num_nodes, num_nodes):
            adj_sem = adj_sem.unsqueeze(1)
        elif adj_sem.dim() == 2 and adj_sem.shape == (num_nodes, num_nodes):
            adj_sem = adj_sem.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, num_nodes, num_nodes)
        else:
            raise ValueError(f"Expected adj_sem [B, N, N] or [N, N], got {tuple(adj_sem.shape)}")

        out = self.propagator(z, adj_phys, adj_wind, adj_sem)
        return out[:, 0, :, :] if out.dim() == 4 else out


class MultiGraphAdaptivePropagation(nn.Module):
    """Propagate a single shared hidden state over future graph views for each horizon."""

    def __init__(
        self,
        channels: int,
        num_horizons: int,
        propagation_steps: int = 3,
        alpha: float = 0.1,
        dropout: float = 0.1,
        use_refinement: bool = True,
    ):
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if num_horizons <= 0:
            raise ValueError(f"num_horizons must be > 0, got {num_horizons}")
        if propagation_steps <= 0:
            raise ValueError(f"propagation_steps must be > 0, got {propagation_steps}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")

        self.channels = channels
        self.num_horizons = num_horizons
        self.propagation_steps = propagation_steps
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.use_refinement = use_refinement

        self.horizon_embeddings = nn.Parameter(torch.randn(num_horizons, channels))
        self.gate = HorizonAwarePropagationGate(channels, channels)
        self.refiner = CurrentStateRefinement(channels, hidden_dim=4 * channels, dropout=dropout) if use_refinement else None

        # Channel-wise normalization used only for gate generation.
        self.message_norm_phys = nn.LayerNorm(channels)
        self.message_norm_wind = nn.LayerNorm(channels)
        self.message_norm_sem = nn.LayerNorm(channels)

    def _normalise_adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        if adj.dim() != 3:
            raise ValueError(f"Expected adjacency [B, N, N], got {tuple(adj.shape)}")
        denom = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return adj / denom

    def forward(
        self,
        z_current: torch.Tensor,
        a_phys: torch.Tensor,
        a_wind_hat: torch.Tensor,
        a_sem_hat: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate the current latent state over future graph views.

        Returns:
            [B, H, N, C]
        """
        # Validate current state.
        if z_current.dim() != 3:
            raise ValueError(f"Expected z_current with shape [B, N, C], got {tuple(z_current.shape)}")
        batch_size, num_nodes, channels = z_current.shape
        if channels != self.channels:
            raise ValueError(
                f"Expected z_current channel dimension C={self.channels}, got {channels} "
                f"from shape {tuple(z_current.shape)}"
            )

        # Validate static physics adjacency.
        if a_phys.dim() != 2:
            raise ValueError(f"Expected a_phys with shape [N, N], got {tuple(a_phys.shape)}")
        if a_phys.shape != (num_nodes, num_nodes):
            raise ValueError(
                f"Expected a_phys shape [{num_nodes}, {num_nodes}], got {tuple(a_phys.shape)}"
            )

        # Validate future adjacencies.
        if isinstance(a_wind_hat, list):
            if len(a_wind_hat) != self.num_horizons:
                raise ValueError(f"Expected {self.num_horizons} sparse wind horizons, got {len(a_wind_hat)}")
            dense_wind = []
            for edge_index, edge_weight in a_wind_hat:
                adj = torch.zeros(batch_size, num_nodes, num_nodes, device=z_current.device, dtype=z_current.dtype)
                if edge_index.numel() > 0:
                    src = edge_index[0]
                    dst = edge_index[1]
                    for batch_idx in range(batch_size):
                        offset = batch_idx * num_nodes
                        mask = (src >= offset) & (src < offset + num_nodes) & (dst >= offset) & (dst < offset + num_nodes)
                        if mask.any():
                            adj[batch_idx, dst[mask] - offset, src[mask] - offset] = edge_weight[mask]
                dense_wind.append(adj)
            a_wind_hat = torch.stack(dense_wind, dim=1)

        if isinstance(a_sem_hat, list):
            if len(a_sem_hat) != self.num_horizons:
                raise ValueError(f"Expected {self.num_horizons} sparse semantic horizons, got {len(a_sem_hat)}")
            dense_sem = []
            for edge_index, edge_weight in a_sem_hat:
                adj = torch.zeros(batch_size, num_nodes, num_nodes, device=z_current.device, dtype=z_current.dtype)
                if edge_index.numel() > 0:
                    src = edge_index[0]
                    dst = edge_index[1]
                    for batch_idx in range(batch_size):
                        offset = batch_idx * num_nodes
                        valid = (src >= offset) & (src < offset + num_nodes) & (dst >= offset) & (dst < offset + num_nodes)
                        if valid.any():
                            src_local = src[valid] - offset
                            dst_local = dst[valid] - offset
                            adj[batch_idx, dst_local, src_local] = edge_weight[valid]
                dense_sem.append(adj)
            a_sem_hat = torch.stack(dense_sem, dim=1)

        if a_wind_hat.dim() != 4:
            raise ValueError(f"Expected a_wind_hat with shape [B, H, N, N], got {tuple(a_wind_hat.shape)}")
        if a_sem_hat.dim() != 4:
            raise ValueError(f"Expected a_sem_hat with shape [B, H, N, N], got {tuple(a_sem_hat.shape)}")

        if a_wind_hat.shape[0] != batch_size:
            raise ValueError(
                f"Expected a_wind_hat batch dimension B={batch_size}, got {a_wind_hat.shape[0]} "
                f"from shape {tuple(a_wind_hat.shape)}"
            )
        if a_sem_hat.shape[0] != batch_size:
            raise ValueError(
                f"Expected a_sem_hat batch dimension B={batch_size}, got {a_sem_hat.shape[0]} "
                f"from shape {tuple(a_sem_hat.shape)}"
            )

        if a_wind_hat.shape[1] != self.num_horizons:
            raise ValueError(
                f"Expected a_wind_hat horizon dimension H={self.num_horizons}, got {a_wind_hat.shape[1]} "
                f"from shape {tuple(a_wind_hat.shape)}"
            )
        if a_sem_hat.shape[1] != self.num_horizons:
            raise ValueError(
                f"Expected a_sem_hat horizon dimension H={self.num_horizons}, got {a_sem_hat.shape[1]} "
                f"from shape {tuple(a_sem_hat.shape)}"
            )

        if a_wind_hat.shape[2:] != (num_nodes, num_nodes):
            raise ValueError(
                f"Expected a_wind_hat spatial dims [{num_nodes}, {num_nodes}], got {tuple(a_wind_hat.shape[2:])} "
                f"from shape {tuple(a_wind_hat.shape)}"
            )
        if a_sem_hat.shape[2:] != (num_nodes, num_nodes):
            raise ValueError(
                f"Expected a_sem_hat spatial dims [{num_nodes}, {num_nodes}], got {tuple(a_sem_hat.shape[2:])} "
                f"from shape {tuple(a_sem_hat.shape)}"
            )

        z_refined = self.refiner(z_current) if self.use_refinement and self.refiner is not None else z_current

        horizon_emb = self.horizon_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        a_phys_norm = self._normalise_adjacency(a_phys.unsqueeze(0).expand(batch_size, -1, -1))

        outputs = []
        for horizon_idx in range(self.num_horizons):
            h = z_refined
            h0 = z_refined
            a_wind_h = self._normalise_adjacency(a_wind_hat[:, horizon_idx, :, :])
            a_sem_h = self._normalise_adjacency(a_sem_hat[:, horizon_idx, :, :])

            for _ in range(self.propagation_steps):
                # Message tensors are normalized only for gate generation.
                m_phys = torch.einsum('bij,bjc->bic', a_phys_norm, h)
                m_wind = torch.einsum('bij,bjc->bic', a_wind_h, h)
                m_sem = torch.einsum('bij,bjc->bic', a_sem_h, h)

                m_phys_gate = self.message_norm_phys(m_phys)
                m_wind_gate = self.message_norm_wind(m_wind)
                m_sem_gate = self.message_norm_sem(m_sem)

                g_phys, g_wind, g_sem = self.gate(
                    h,
                    m_phys_gate,
                    m_wind_gate,
                    m_sem_gate,
                    horizon_emb[:, horizon_idx, :],
                )

                # Preserve the original message tensors for the final gated fusion.
                m = g_phys * m_phys + g_wind * m_wind + g_sem * m_sem
                m = self.dropout(m)
                h = (1.0 - self.alpha) * m + self.alpha * h0

            outputs.append(h)

        return torch.stack(outputs, dim=1)