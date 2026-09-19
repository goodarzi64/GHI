import torch
import torch.nn as nn


class ForecastHead(nn.Module):
    """Lightweight regression head for horizon-wise node forecasts."""

    def __init__(self, channels: int, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = 2 * channels if hidden_dim is None else hidden_dim
        self.norm = nn.LayerNorm(channels)
        self.proj_in = nn.Linear(channels, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj_out = nn.Linear(hidden_dim, 1)

    def forward(self, h_final: torch.Tensor) -> torch.Tensor:
        """Map [B, H, N, C] to [B, H, N]."""
        if h_final.dim() != 4:
            raise ValueError(f"Expected propagated states [B, H, N, C], got {tuple(h_final.shape)}")

        x = self.norm(h_final)
        batch_size, num_horizons, num_nodes, channels = h_final.shape
        x = x.reshape(batch_size * num_horizons * num_nodes, channels)
        x = self.proj_in(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.proj_out(x).squeeze(-1)
        return x.reshape(batch_size, num_horizons, num_nodes)


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