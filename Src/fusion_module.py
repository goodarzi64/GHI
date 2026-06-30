"""
Fusion Module for Multi-Graph Spatial Embeddings

Combines outputs from three GATv2 spatial encoders (static, semantic, wind graphs)
into a single unified spatial representation via learned fusion weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class AttentionFusion(nn.Module):
    """
    Attention-based fusion of multi-graph embeddings.
    
    Learns context-aware importance weights for each graph type.
    Instead of scoring each embedding independently, the gate first sees
    the full concatenation of all graph embeddings per node.
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
        
        # Optional: learnable temperature for controlling focus
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-graph embeddings via attention weights.
        
        Args:
            embeddings: List of G embeddings, each [N, F] or [B, N, F]
        
        Returns:
            Fused embedding [N, F] or [B, N, F]
        """
        if len(embeddings) != self.n_graphs:
            raise ValueError(f"Expected {self.n_graphs} embeddings, got {len(embeddings)}")

        # Joint context: each graph sees the others before scoring.
        joint = torch.cat(embeddings, dim=-1)  # [N, G*F] or [B, N, G*F]
        scores = self.attention(joint)  # [N, G] or [B, N, G]
        
        alpha = F.softmax(scores / self.temperature, dim=-1)
        
        # Weighted sum
        fused = torch.zeros_like(embeddings[0])
        for i, emb in enumerate(embeddings):
            fused = fused + alpha[..., i:i+1] * emb
        
        return fused, alpha


class ConcatFusion(nn.Module):
    """
    Concatenation-based fusion with projection.
    
    Concatenates embeddings from all graphs and projects to target dimension.
    """
    
    def __init__(self, embedding_dim: int, n_graphs: int = 3, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_graphs = n_graphs
        
        # Project from [n_graphs * embedding_dim] back to [embedding_dim]
        self.projection = nn.Sequential(
            nn.Linear(n_graphs * embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )
    
    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse by concatenation and projection.
        
        Args:
            embeddings: List of G embeddings, each [N, F] or [B, N, F]
        
        Returns:
            Fused embedding [N, F] or [B, N, F]
        """
        concat = torch.cat(embeddings, dim=-1)
        fused = self.projection(concat)
        return fused


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
    
    Supports multiple fusion strategies: attention, concatenation, or learned weights.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        n_graphs: int = 3,
        fusion_method: str = 'attention',  # 'attention', 'concat', 'learned'
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_graphs = n_graphs
        self.fusion_method = fusion_method
        
        if fusion_method == 'attention':
            self.fusion = AttentionFusion(embedding_dim, n_graphs, dropout)
        elif fusion_method == 'concat':
            self.fusion = ConcatFusion(embedding_dim, n_graphs, dropout)
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


class MultiScaleFusion(nn.Module):
    """
    Multi-scale fusion: applies fusion at multiple spatial scales,
    then combines results.
    
    Useful when embeddings have different scales or when hierarchical
    fusion is desired.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        n_graphs: int = 3,
        n_scales: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_graphs = n_graphs
        self.n_scales = n_scales
        
        # One fusion module per scale
        self.fusions = nn.ModuleList([
            SpatialFusionModule(embedding_dim, n_graphs, 'attention', dropout)
            for _ in range(n_scales)
        ])
        
        # Final scale aggregation
        self.scale_fusion = SpatialFusionModule(embedding_dim, n_scales, 'learned', dropout)
    
    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply multi-scale fusion.
        
        Args:
            embeddings: List of G embeddings
        
        Returns:
            Multi-scale fused embedding
        """
        scale_outputs = []
        for fusion in self.fusions:
            out = fusion(embeddings)
            scale_outputs.append(out)
        
        # Fuse across scales
        final = self.scale_fusion(scale_outputs)
        return final


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
        fusion_type: 'attention', 'concat', 'learned', 'multiscale'
        **kwargs: Additional arguments passed to the module
    
    Returns:
        Initialized fusion module
    """
    if fusion_type == 'multiscale':
        return MultiScaleFusion(embedding_dim, n_graphs, **kwargs)
    else:
        return SpatialFusionModule(embedding_dim, n_graphs, fusion_type, **kwargs)
