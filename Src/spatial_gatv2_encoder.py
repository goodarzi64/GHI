"""
Spatial GATv2 Encoder for Multi-Graph Spatio-Temporal Forecasting

Three graph types (static, dynamic, wind) each processed by a GATv2 layer with
edge weights applied as bias (beta * log(A)) to attention logits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import softmax
from typing import Tuple, Optional
from torch import Tensor
from torch.nn import Parameter
from typing import Optional, Tuple, Union
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

    Attention logits:

        e_ij =
            a^T LeakyReLU(
                W_s x_i + W_t x_j
            )
            + beta * log(A_ij)

    Attention weights:

        alpha_ij = softmax(e_ij)

    where:
        A_ij : adjacency weight in [0,1]
        beta : learnable scalar

    The adjacency does NOT pass through a learnable edge encoder.
    It directly biases attention logits.
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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.fill_value = fill_value
        self.share_weights = share_weights

        # -------------------------
        # Node projections
        # -------------------------

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

        # Attention vector
        self.att = Parameter(
            torch.empty(1, heads, out_channels)
        )

        # Structural bias strength
        self.beta = Parameter(
            torch.ones(1)
        )

        total_out_channels = (
            heads * out_channels
            if concat
            else out_channels
        )

        if bias:
            self.bias = Parameter(
                torch.empty(total_out_channels)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):

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

        H = self.heads
        C = self.out_channels

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

        if self.concat:
            out = out.view(
                -1,
                H * C,
            )
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
        """
        Compute attention coefficients:

            e_ij =
                a^T LeakyReLU(W_s x_i + W_t x_j)
                + beta * (log(A_ij) - mu_log)

        where

            mu_log = mean(log(A))

        over all existing edges.

        This makes edges stronger than the graph geometric mean
        receive positive bias and weaker edges receive negative bias.
        """

        # --------------------------------------------------
        # Standard GATv2 attention logits
        # --------------------------------------------------
        x = x_i + x_j

        x = F.leaky_relu(
            x,
            negative_slope=self.negative_slope,
        )

        logits = (x * self.att).sum(dim=-1)

        # --------------------------------------------------
        # Structural bias
        # b_ij = beta * (log(A_ij) - mu_log)
        # --------------------------------------------------
        if edge_weight is not None:

            edge_weight = edge_weight.clamp(min=1e-8)

            log_edge_weight = torch.log(edge_weight)

            # global mean over existing edges
            mu_log = log_edge_weight.mean()

            bias_term = self.beta * (
                log_edge_weight - mu_log
            )

            logits = logits + bias_term.unsqueeze(-1)

        # --------------------------------------------------
        # Neighborhood normalization
        # --------------------------------------------------
        alpha = softmax(
            logits,
            index,
            ptr,
            dim_size,
        )

        alpha = F.dropout(
            alpha,
            p=self.dropout,
            training=self.training,
        )

        return alpha

    def message(
        self,
        x_j: Tensor,
        alpha: Tensor,
    ) -> Tensor:

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
    Multi-graph spatial encoder processing 3 graph types (static, dynamic, wind)
    separately via StructuralBiasGATv2 layers, producing node embeddings per graph.
    
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
