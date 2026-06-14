"""
Spatial Encoder + Fusion Integration Example

Demonstrates how to use GATv2EdgeBiasConv spatial encoder and fusion module
with the three graph types (static, dynamic, wind) in a Colab environment.

For Colab setup, install dependencies in first cell:
    !pip install torch torch-geometric torch-geometric-temporal pyproj scikit-learn
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional

from spatial_gatv2_encoder import SpatialGATv2EncoderBatched
from fusion_module import SpatialFusionModule
from temporal_conv_module import TemporalConvModule

# These utilities can still be used if imported from the same directory
# from Graph_build import build_static_adjacency, build_semantic_adjacency, WindAdjacency


class SpatialEncoderWithFusion(nn.Module):
    """
    Complete spatial encoder pipeline:
    1. Takes node features [B, N, F_in]
    2. Processes through 3 separate GATv2 encoders (one per graph type)
    3. Fuses outputs into unified spatial representation [B, N, F_spatial]
    """
    
    def __init__(
        self,
        node_in_features: int,
        spatial_out_features: int = 64,
        gatv2_heads: int = 8,
        fusion_method: str = 'attention',
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.node_in_features = node_in_features
        self.spatial_out_features = spatial_out_features
        
        # GATv2 encoders for 3 graph types
        self.spatial_encoder = SpatialGATv2EncoderBatched(
            in_features=node_in_features,
            out_features=spatial_out_features,
            n_graphs=3,  # static, dynamic, wind
            heads=gatv2_heads,
            dropout=dropout,
        )
        
        # Fusion module
        self.fusion = SpatialFusionModule(
            embedding_dim=spatial_out_features,
            n_graphs=3,
            fusion_method=fusion_method,
            dropout=dropout,
        )
    
    def forward(
        self,
        x_batch: torch.Tensor,
        edge_indices: List[torch.Tensor],
        edge_weights: List[Optional[torch.Tensor]],
        return_per_graph: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_batch: Node features [B, N, F_in]
            edge_indices: List of 3 edge_index tensors [2, E_i]
            edge_weights: List of 3 edge_weight tensors (or None for dense)
            return_per_graph: If True, also return per-graph embeddings
        
        Returns:
            Fused spatial embedding [B, N, F_spatial]
            optionally: (fused, [emb_static, emb_dynamic, emb_wind])
        """
        # Encode via GATv2 per graph
        embeddings = self.spatial_encoder(x_batch, edge_indices, edge_weights)
        
        # Fuse
        spatial_repr = self.fusion(embeddings)
        
        if return_per_graph:
            return spatial_repr, embeddings
        return spatial_repr


class SpatialTemporalForecastingModel(nn.Module):
    """
    Spatial-temporal pipeline that applies spatial fusion at each time step,
    then fuses the resulting sequence with temporal diurnal and seasonal kernels.
    """

    def __init__(
        self,
        node_in_features: int,
        spatial_out_features: int = 64,
        gatv2_heads: int = 8,
        fusion_method: str = 'attention',
        temporal_hidden_dim: int = 64,
        temporal_output_dim: Optional[int] = None,
        diurnal_window: int = 24,
        seasonal_window: int = 168,
        seasonal_dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spatial_encoder = SpatialEncoderWithFusion(
            node_in_features=node_in_features,
            spatial_out_features=spatial_out_features,
            gatv2_heads=gatv2_heads,
            fusion_method=fusion_method,
            dropout=dropout,
        )
        self.temporal_module = TemporalConvModule(
            node_feature_dim=spatial_out_features,
            hidden_dim=temporal_hidden_dim,
            output_dim=temporal_output_dim,
            diurnal_window=diurnal_window,
            seasonal_window=seasonal_window,
            seasonal_dilation=seasonal_dilation,
            dropout=dropout,
            use_global_branch=True,
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        edge_indices: List[torch.Tensor],
        edge_weights: List[Optional[torch.Tensor]],
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x_seq: Input sequence [B, W, N, F_in]
            edge_indices: List of 3 edge_index tensors
            edge_weights: List of 3 edge_weight tensors
            return_attention: If True, return temporal attention weights

        Returns:
            Temporal fused output [B, N, output_dim]
        """
        B, W, N, F_in = x_seq.shape
        spatial_sequence = []
        for t in range(W):
            x_t = x_seq[:, t, :, :]
            spatial_repr = self.spatial_encoder(x_t, edge_indices, edge_weights)
            spatial_sequence.append(spatial_repr)

        spatial_sequence = torch.stack(spatial_sequence, dim=1)

        if return_attention:
            fused, attention_weights = self.temporal_module(spatial_sequence, return_attention=True)
            return fused, attention_weights
        return self.temporal_module(spatial_sequence)


def prepare_adjacency_matrices(
    df_geo: Optional = None,
    ghi_history: Optional[torch.Tensor] = None,
    tcc_history: Optional[torch.Tensor] = None,
    wind_features: Optional[torch.Tensor] = None,
    device: str = 'cpu',
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Helper to build adjacency matrices for 3 graph types.
    
    Requires:
    - df_geo: DataFrame with 'latitude' and 'longitude' columns for static graph
    - ghi_history: [N, W] historical GHI for semantic graph
    - tcc_history: [N, W] historical TCC for semantic graph
    - wind_features: [N, F_wind] wind features for wind kernel
    
    Returns:
        (edge_indices, edge_weights): Lists of length 3 for [static, dynamic, wind]
    """
    try:
        from Graph_build import (
            build_static_adjacency,
            build_semantic_adjacency,
            WindAdjacency,
        )
    except ImportError:
        raise ImportError("Graph_build module not found.")
    
    edge_indices = []
    edge_weights = []
    
    # 1. Static graph (geographic distance)
    if df_geo is not None:
        static_adj = build_static_adjacency(df_geo=df_geo, device=device, k=5)
        A_static = static_adj['A_sym_norm']  # Use symmetric normalized
    else:
        # Fallback: fully connected
        N = 10  # placeholder
        A_static = torch.ones(N, N, device=device) / N
    
    # Dense to sparse
    from torch_sparse import dense_to_sparse
    ei, ew = dense_to_sparse(A_static)
    edge_indices.append(ei)
    edge_weights.append(ew)
    
    # 2. Semantic graph (DTW distance on GHI+TCC history)
    if ghi_history is not None and tcc_history is not None:
        semantic_adj = build_semantic_adjacency(
            ghi=ghi_history.to(device),
            tcc=tcc_history.to(device),
            k=5,
        )
        A_semantic = semantic_adj['A_sym_norm']
    else:
        A_semantic = torch.ones_like(A_static) / A_static.shape[0]
    
    ei, ew = dense_to_sparse(A_semantic)
    edge_indices.append(ei)
    edge_weights.append(ew)
    
    # 3. Wind graph (spatiotemporal wind-based adjacency)
    if wind_features is not None:
        # Build distance/bearing matrices from geo if available
        if df_geo is not None:
            from Graph_build import build_geo_matrices
            geo_mats = build_geo_matrices(df_geo, device=device)
            D_ij = geo_mats['dist_matrix']
            Theta_ij = geo_mats['theta_matrix']
        else:
            N = wind_features.shape[0]
            D_ij = torch.ones(N, N, device=device)
            Theta_ij = torch.zeros(N, N, device=device)
        
        wind_kernel = WindAdjacency(D_ij, Theta_ij, device=device)
        A_wind = wind_kernel(wind_features[None, ...], sparse=False, k=5)  # [1, N, N]
        A_wind = A_wind[0]  # [N, N]
    else:
        A_wind = torch.ones_like(A_static) / A_static.shape[0]
    
    ei, ew = dense_to_sparse(A_wind)
    edge_indices.append(ei)
    edge_weights.append(ew)
    
    return edge_indices, edge_weights


# Example usage in Colab:
if __name__ == '__main__':
    """
    Minimal example showing how to use in Colab.
    """
    print("Example: Spatial GATv2 Encoder with Fusion")
    
    # Hyperparameters
    B, N, F_in = 4, 20, 8  # batch_size, num_nodes, node_features
    F_spatial = 64
    
    # Create module
    encoder = SpatialEncoderWithFusion(
        node_in_features=F_in,
        spatial_out_features=F_spatial,
        gatv2_heads=4,
        fusion_method='attention',
    )
    
    # Create dummy data
    x_batch = torch.randn(B, N, F_in)
    
    # Create dummy adjacency (sparse format)
    # In real usage, use prepare_adjacency_matrices() above
    edge_indices = [
        torch.LongTensor([[0, 1, 2], [1, 2, 3]]),  # static
        torch.LongTensor([[0, 2, 3], [2, 3, 4]]),  # dynamic
        torch.LongTensor([[1, 2, 4], [2, 4, 5]]),  # wind
    ]
    
    edge_weights = [
        torch.ones(3),  # static weights
        torch.ones(3),  # dynamic weights
        torch.ones(3),  # wind weights
    ]
    
    # Forward pass
    spatial_repr = encoder(x_batch, edge_indices, edge_weights)
    print(f"Input shape: {x_batch.shape}")
    print(f"Spatial representation shape: {spatial_repr.shape}")
    print(f"Expected: [{B}, {N}, {F_spatial}]")
    
    # With per-graph outputs
    spatial_repr, per_graph = encoder(x_batch, edge_indices, edge_weights, return_per_graph=True)
    print(f"\nPer-graph embeddings:")
    for i, emb in enumerate(per_graph):
        print(f"  Graph {i}: {emb.shape}")

    # Temporal fusion example using the spatial-temporal pipeline
    W = 48
    x_seq = torch.randn(B, W, N, F_in)
    spatio_temporal_model = SpatialTemporalForecastingModel(
        node_in_features=F_in,
        spatial_out_features=F_spatial,
        gatv2_heads=4,
        fusion_method='attention',
        temporal_hidden_dim=64,
        temporal_output_dim=64,
        diurnal_window=24,
        seasonal_window=168,
        seasonal_dilation=1,
        dropout=0.1,
    )
    temporal_output = spatio_temporal_model(x_seq, edge_indices, edge_weights)
    print(f"Temporal output shape: {temporal_output.shape}")
    print(f"Expected: [{B}, {N}, 64]")
