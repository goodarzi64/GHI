import torch
import torch.nn as nn


class ForecastHead(nn.Module):
    """Lightweight regression head for horizon-wise node forecasts.

    Supports both the current maintained interface, where the input is a 4D tensor
    [B, H, N, C], and the legacy decoder-style interface that passes a current
    state [B, N, C] together with explicit horizon indices.
    """

    def __init__(
        self,
        channels: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        num_horizons: int | None = None,
        horizon_emb_dim: int | None = None,
    ):
        super().__init__()
        hidden_dim = 2 * channels if hidden_dim is None else hidden_dim
        self.num_horizons = num_horizons
        self.horizon_emb_dim = horizon_emb_dim or 0
        self.norm = nn.LayerNorm(channels)
        self.proj_in = nn.Linear(channels, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj_out = nn.Linear(hidden_dim, 1)

        if num_horizons is not None and num_horizons > 0:
            self.horizon_embeddings = nn.Parameter(torch.randn(num_horizons, channels))
        else:
            self.horizon_embeddings = None

    def forward(self, h_final: torch.Tensor, horizon_idx: torch.Tensor | None = None) -> torch.Tensor:
        """Map state(s) to [B, H, N]. Accepts either [B, H, N, C] or [B, N, C]."""
        if h_final.dim() == 4:
            x = self.norm(h_final)
            batch_size, num_horizons, num_nodes, channels = h_final.shape
            x = x.reshape(batch_size * num_horizons * num_nodes, channels)
            x = self.proj_in(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.proj_out(x).squeeze(-1)
            return x.reshape(batch_size, num_horizons, num_nodes)

        if h_final.dim() == 3:
            batch_size, num_nodes, channels = h_final.shape
            if self.num_horizons is not None and self.num_horizons > 0:
                horizon_count = self.num_horizons
            elif horizon_idx is not None:
                horizon_count = int(torch.as_tensor(horizon_idx).max().item()) + 1
            else:
                horizon_count = 1

            state = self.norm(h_final).unsqueeze(1).expand(batch_size, horizon_count, num_nodes, channels)
            state = state.reshape(batch_size * horizon_count * num_nodes, channels)
            state = self.proj_in(state)
            state = self.act(state)
            state = self.drop(state)
            pred = self.proj_out(state).squeeze(-1)
            return pred.reshape(batch_size, horizon_count, num_nodes)

        raise ValueError(f"Expected state tensor of rank 3 or 4, got {tuple(h_final.shape)}")


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