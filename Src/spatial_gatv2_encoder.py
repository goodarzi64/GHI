"""
Spatial GATv2 Encoder for Multi-Graph Spatio-Temporal Forecasting.

This module defines a structural-bias variant of GATv2 that injects
adjacency weights directly into the attention logits and two encoder
wrappers for single-graph and batched multi-graph inputs.

Input and output dimensionality:
- Single-graph encoder: expects node features of shape [N, F_in] and returns
  embeddings of shape [N, F_out].
- Batched encoder: expects node features of shape [B, N, F_in] and returns
  embeddings of shape [B, N, F_out].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import softmax
from typing import Tuple, Optional, Union
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
)
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    softmax,
)
class StructuralBiasGATv2(MessagePassing):
    """
    GATv2 with structural adjacency bias.

    The layer computes standard GATv2 attention logits and then adds a
    learnable bias term derived from the adjacency weights.
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 4,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        fill_value: str = "mean",
        bias: bool = True,
        share_weights: bool = False,
    ):
        super().__init__(aggr="add", node_dim=0)

        # Layer configuration
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.fill_value = fill_value
        self.share_weights = share_weights

        # --------------------------------------------------
        # Node input projections
        # --------------------------------------------------
        # Project left and right node features into the attention space.
        if isinstance(in_channels, int):
            self.lin_l = Linear(
                in_channels,
                heads * out_channels,
                bias=False,
                weight_initializer="glorot",
            )

            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(
                    in_channels,
                    heads * out_channels,
                    bias=False,
                    weight_initializer="glorot",
                )
        else:
            self.lin_l = Linear(
                in_channels[0],
                heads * out_channels,
                bias=False,
                weight_initializer="glorot",
            )

            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(
                    in_channels[1],
                    heads * out_channels,
                    bias=False,
                    weight_initializer="glorot",
                )

        # Attention vector for each head
        self.att = Parameter(torch.empty(1, heads, out_channels))

        # Learnable scalar controlling adjacency bias strength
        self.beta = Parameter(torch.ones(1))

        total_out_channels = heads * out_channels if concat else out_channels

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights and bias parameters."""
        self.lin_l.reset_parameters()

        if self.lin_r is not self.lin_l:
            self.lin_r.reset_parameters()

        glorot(self.att)
        zeros(self.bias)

        with torch.no_grad():
            self.beta.fill_(1.0)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_weight: Optional[Tensor] = None,
    ):
        """Forward pass through the structurally biased GATv2 layer."""

        H = self.heads
        C = self.out_channels

        # --------------------------------------------------
        # Project node features into attention space
        # --------------------------------------------------
        if isinstance(x, Tensor):
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x
            x_l = self.lin_l(x_l).view(-1, H, C)
            x_r = self.lin_r(x_r).view(-1, H, C)

        # --------------------------------------------------
        # Compute attention coefficients and propagate
        # --------------------------------------------------
        alpha = self.edge_updater(
            edge_index,
            x=(x_l, x_r),
            edge_weight=edge_weight,
        )

        out = self.propagate(
            edge_index,
            x=(x_l, x_r),
            alpha=alpha,
        )

        # --------------------------------------------------
        # Merge multi-head outputs
        # --------------------------------------------------
        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def edge_update(
        self,
        x_j: Tensor,
        x_i: Tensor,
        edge_weight: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        dim_size: Optional[int],
    ) -> Tensor:
        """Compute the attention weights for each edge."""

        # --------------------------------------------------
        # Standard GATv2 attention logits
        # --------------------------------------------------
        x = x_i + x_j
        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        logits = (x * self.att).sum(dim=-1)

        # --------------------------------------------------
        # Structural adjacency bias
        # Add a learnable term based on log(A_ij) centered by the mean.
        # --------------------------------------------------
        if edge_weight is not None:
            edge_weight = edge_weight.clamp(min=1e-8)
            log_edge_weight = torch.log(edge_weight)
            mu_log = log_edge_weight.mean()

            bias_term = self.beta * (log_edge_weight - mu_log)
            logits = logits + bias_term.unsqueeze(-1)

        # --------------------------------------------------
        # Normalize attention logits across each node's neighbors
        # --------------------------------------------------
        alpha = softmax(logits, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(
        self,
        x_j: Tensor,
        alpha: Tensor,
    ) -> Tensor:
        """Message function for propagation: scale source node features by attention."""
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"{self.in_channels}, "
            f"{self.out_channels}, "
            f"heads={self.heads})"
        )



class SpatialGATv2Encoder(nn.Module):
    """
    Multi-graph spatial encoder using separate StructuralBiasGATv2 layers.

    Each of the three graph types (static, dynamic, wind) is encoded
    independently and returns a list of graph-specific embeddings.
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

        # Encoder configuration
        self.in_features = in_features
        self.out_features = out_features
        self.n_graphs = n_graphs  # static, dynamic, wind
        self.use_bias_scaling = use_bias_scaling

        # Create one SpatialGATv2 encoder per graph type
        self.encoders = nn.ModuleList([
            StructuralBiasGATv2(
                in_features,
                out_features,
                heads=heads,
                concat=True,
                negative_slope=0.2,
                dropout=dropout,
                add_self_loops=True,
                fill_value="mean",
                bias=True,
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
        Forward pass for the multi-graph encoder.

        Each graph encoder receives the same node feature matrix but different
        adjacency information, producing separate embeddings for each graph.
        """
        embeddings = []
        for i, encoder in enumerate(self.encoders):
            emb = encoder(x, edge_indices[i], edge_weights[i] if edge_weights[i] is not None else None)
            embeddings.append(emb)

        return embeddings


class SpatialGATv2EncoderBatched(nn.Module):
    """
    Batched spatial encoder for inputs shaped [B, N, F_in].

    This wrapper flattens the batch dimension for the GATv2 encoders and
    reconstructs batched embeddings after propagation.
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

        # One StructuralBiasGATv2 encoder per graph type
        self.encoders = nn.ModuleList([
            StructuralBiasGATv2(
                in_features,
                out_features,
                heads=heads,
                concat=True,
                negative_slope=0.2,
                dropout=dropout,
                add_self_loops=True,
                fill_value="mean",
                bias=True,
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
        Process batched node features through the graph encoders.

        The batch dimension is flattened for propagation, then reshaped back
        into [B, N, out_features] after the attention update.
        """
        B, N, F_in = x_batch.shape
        embeddings = []

        for i, encoder in enumerate(self.encoders):
            # Flatten batch for the graph operation
            x_flat = x_batch.reshape(B * N, F_in)

            edge_weight_i = edge_weights[i]
            if edge_weight_i is not None:
                if edge_weight_i.dim() == 3:  # Batched edge weights [B, N, N]
                    edge_weight_flat = edge_weight_i[0].flatten()
                else:  # Shared edge weights [N, N]
                    edge_weight_flat = edge_weight_i.flatten()
            else:
                edge_weight_flat = None

            emb_flat = encoder(x_flat, edge_indices[i], edge_weight_flat)
            emb_batch = emb_flat.reshape(B, N, -1)
            embeddings.append(emb_batch)

        return embeddings
