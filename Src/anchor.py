import torch
import torch.nn as nn


class CurrentStateAnchor(nn.Module):
    """Anchor a pooled temporal state with its most recent spatial embedding."""

    def __init__(self, spatial_dim: int, latent_dim: int):
        super().__init__()
        self.proj = nn.Linear(spatial_dim, latent_dim)
        self.gate = nn.Linear(spatial_dim + latent_dim, latent_dim)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, z_last: torch.Tensor, s_last: torch.Tensor) -> torch.Tensor:
        """Return an anchored state [B, N, latent_dim]."""
        if z_last.dim() != 3 or s_last.dim() != 3:
            raise ValueError("Expected z_last [B, N, C] and s_last [B, N, S]")
        if z_last.shape[:2] != s_last.shape[:2]:
            raise ValueError("z_last and s_last must have matching batch and node dimensions")
        gate = torch.sigmoid(self.gate(torch.cat([s_last, z_last], dim=-1)))
        return z_last + gate * self.proj(s_last)
