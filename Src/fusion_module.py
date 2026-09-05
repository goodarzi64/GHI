"""
Fusion Module for Multi-Graph Spatial Embeddings

Combines outputs from three GATv2 spatial encoders (static, semantic, wind graphs)
into a single unified spatial representation via learned fusion weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class NodeAdaptiveMultiGraphFusion(nn.Module):
    """
    Node-adaptive attention fusion for multi-graph embeddings.

    Learns context-aware, node-wise importance weights for each graph type.
    The temperature parameter is learnable but reparameterized via softplus
    to ensure it is always positive and numerically stable.
    """

    def __init__(self, embedding_dim: int, n_graphs: int = 3, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_graphs = n_graphs

        joint_input_dim = embedding_dim * n_graphs
        hidden_dim = max(embedding_dim // 2, n_graphs)

        # Context-aware scoring network: [G * F] -> [G]
        self.attention = nn.Sequential(
            nn.Linear(joint_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_graphs),
        )

        # Learnable raw temperature parameter (reparameterized with softplus)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self._min_temperature = 1e-3

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-graph embeddings via attention weights.

        Args:
            embeddings: List of G embeddings, each [N, F] or [B, N, F]

        Returns:
            (fused, alpha): fused embedding and attention weights
        """
        if len(embeddings) != self.n_graphs:
            raise ValueError(f"Expected {self.n_graphs} embeddings, got {len(embeddings)}")

        # Joint context: each graph sees the others before scoring.
        joint = torch.cat(embeddings, dim=-1)  # [N, G*F] or [B, N, G*F]
        scores = self.attention(joint)  # [N, G] or [B, N, G]

        temp = F.softplus(self.temperature) + self._min_temperature
        alpha = F.softmax(scores / temp, dim=-1)

        # Weighted sum
        fused = torch.zeros_like(embeddings[0])
        for i, emb in enumerate(embeddings):
            fused = fused + alpha[..., i:i+1] * emb

        return fused, alpha


class LearnedWeightFusion(nn.Module):
    """
    Learnable fixed weights for each graph (simpler than attention).
    
    fusion = sum(w_i * z_i) where w_i are learnable scalars.
    """
    
    def __init__(self, n_graphs: int = 3):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_graphs) / n_graphs)
    
    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse with learnable weights.
        
        Args:
            embeddings: List of G embeddings, each [N, F] or [B, N, F]
        
        Returns:
            Fused embedding [N, F] or [B, N, F]
        """
        alpha = F.softmax(self.weights, dim=0)
        fused = torch.zeros_like(embeddings[0])
        for i, emb in enumerate(embeddings):
            fused = fused + alpha[i] * emb
        return fused


class SpatialFusionModule(nn.Module):
    """
    Complete fusion module that combines embeddings from three GATv2 encoders
    (static, semantic, wind graph types).
    
    Supports multiple fusion strategies: attention or learned weights.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        n_graphs: int = 3,
        fusion_method: str = 'attention',  # 'attention', 'learned'
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_graphs = n_graphs
        self.fusion_method = fusion_method
        
        if fusion_method == 'attention':
            self.fusion = NodeAdaptiveMultiGraphFusion(embedding_dim, n_graphs, dropout)
        elif fusion_method == 'learned':
            self.fusion = LearnedWeightFusion(n_graphs)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward(
        self,
        embeddings: List[torch.Tensor],
        return_weights: bool = False,
    ) -> torch.Tensor:
        """
        Fuse spatial embeddings from multiple graphs.
        
        Args:
            embeddings: List of 3 embeddings [N, F] or [B, N, F]
            return_weights: If True and using attention, return fusion weights
        
        Returns:
            Fused embedding [N, F] or [B, N, F]
            optionally: fusion weights alpha
        """
        if self.fusion_method == 'attention':
            fused, alpha = self.fusion(embeddings)
            if return_weights:
                return fused, alpha
            return fused
        else:
            return self.fusion(embeddings)


# MultiScaleFusion removed — use SpatialFusionModule directly with desired method


def create_fusion_module(
    embedding_dim: int,
    n_graphs: int = 3,
    fusion_type: str = 'attention',
    **kwargs,
) -> nn.Module:
    """
    Factory function to create appropriate fusion module.
    
    Args:
        embedding_dim: Dimension of embeddings to fuse
        n_graphs: Number of graphs / embeddings
        fusion_type: 'attention', 'learned', 'multiscale'
        **kwargs: Additional arguments passed to the module
    
    Returns:
        Initialized fusion module
    """
    if fusion_type not in ('attention', 'learned'):
        raise ValueError(f"Unknown fusion_type: {fusion_type}. Supported: 'attention', 'learned'.")
    return SpatialFusionModule(embedding_dim, n_graphs, fusion_type, **kwargs)
