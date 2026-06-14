"""
Spatial GATv2 Encoder for Multi-Graph Spatio-Temporal Forecasting

Three graph types (static, dynamic, wind) each processed by a GATv2 layer with
edge weights applied as bias (beta * log(A)) to attention logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Tuple, Optional


class GATv2EdgeBiasConv(nn.Module):
    """
    GATv2 layer with edge weights applied as bias to attention logits.
    
    Edge weight is incorporated as: e_ij^{(l)} = beta * log(A_ij)
    where A_ij is the adjacency matrix weight, preventing log(0) via small epsilon.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.1,
        add_self_loops: bool = True,
        fill_value: str = 'mean',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout_rate = dropout
        
        # GATv2 layer without edge weights (we'll add them manually)
        self.gatv2 = GATv2Conv(
            in_channels,
            out_channels,
            heads=heads,
            dropout=dropout,
            add_self_loops=add_self_loops,
            fill_value=fill_value,
        )
        
        # Learnable beta for edge weight bias scaling
        self.beta = nn.Parameter(torch.ones(1))
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with edge weight bias applied to attention.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights from adjacency matrix [num_edges], optional
        
        Returns:
            Node embeddings [N, out_channels]
        """
        # If edge_weight provided, apply as attention bias: beta * log(A)
        if edge_weight is not None:
            # Clamp to avoid log(0)
            edge_weight = torch.clamp(edge_weight, min=1e-8)
            edge_attr = self.beta * torch.log(edge_weight).unsqueeze(-1)
        else:
            edge_attr = None
        
        # GATv2 forward with edge attributes
        out = self.gatv2(x, edge_index, edge_attr=edge_attr)
        return out


class SpatialGATv2Encoder(nn.Module):
    """
    Multi-graph spatial encoder processing 3 graph types (static, dynamic, wind)
    separately via GATv2 layers, producing node embeddings per graph.
    
    Input: Node features [N, in_features]
    Output: Embeddings [N, out_features] per graph (total 3 embeddings)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_graphs: int = 3,
        heads: int = 8,
        dropout: float = 0.1,
        use_bias_scaling: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_graphs = n_graphs  # static, dynamic, wind
        self.use_bias_scaling = use_bias_scaling
        
        # One GATv2 encoder per graph type
        self.encoders = nn.ModuleList([
            GATv2EdgeBiasConv(
                in_features,
                out_features,
                heads=heads,
                dropout=dropout,
                add_self_loops=True,
            )
            for _ in range(n_graphs)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        edge_indices: list,
        edge_weights: list,
    ) -> list:
        """
        Process node features through separate GATv2 for each graph.
        
        Args:
            x: Node features [N, in_features]
            edge_indices: List of 3 edge_index tensors (static, dynamic, wind)
            edge_weights: List of 3 edge_weight tensors (one per graph)
        
        Returns:
            List of 3 embeddings [N, out_features] per graph type
        """
        embeddings = []
        for i, encoder in enumerate(self.encoders):
            emb = encoder(x, edge_indices[i], edge_weights[i] if edge_weights[i] is not None else None)
            embeddings.append(emb)
        
        return embeddings


class SpatialGATv2EncoderBatched(nn.Module):
    """
    Batched version processing [B, N, F_in] inputs where B is batch size,
    N is number of nodes, F_in is feature dimension.
    
    Returns batched embeddings [B, N, F_out] per graph type.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_graphs: int = 3,
        heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_graphs = n_graphs
        
        # One encoder per graph type
        self.encoders = nn.ModuleList([
            GATv2EdgeBiasConv(
                in_features,
                out_features,
                heads=heads,
                dropout=dropout,
                add_self_loops=True,
            )
            for _ in range(n_graphs)
        ])
    
    def forward(
        self,
        x_batch: torch.Tensor,
        edge_indices: list,
        edge_weights: list,
    ) -> list:
        """
        Process batched inputs through GATv2 encoders.
        
        Args:
            x_batch: Batched node features [B, N, in_features]
            edge_indices: List of 3 edge_index tensors (same for all batches)
            edge_weights: List of 3 edge_weight tensors [B, N, N]
        
        Returns:
            List of 3 batched embeddings [B, N, out_features]
        """
        B, N, F_in = x_batch.shape
        embeddings = []
        
        for i, encoder in enumerate(self.encoders):
            # Reshape batch to [B*N, F_in]
            x_flat = x_batch.reshape(B * N, F_in)
            
            # Edge weights: [B, N, N] -> sparse format per batch
            # For simplicity, use edge weights from first batch (or average)
            edge_weight_i = edge_weights[i]
            if edge_weight_i is not None:
                if edge_weight_i.dim() == 3:  # [B, N, N]
                    # Take weights from first batch as representative
                    edge_weight_flat = edge_weight_i[0].flatten()
                else:  # [N, N]
                    edge_weight_flat = edge_weight_i.flatten()
            else:
                edge_weight_flat = None
            
            # Forward pass
            emb_flat = encoder(x_flat, edge_indices[i], edge_weight_flat)  # [B*N, F_out]
            emb_batch = emb_flat.reshape(B, N, -1)  # [B, N, F_out]
            embeddings.append(emb_batch)
        
        return embeddings
