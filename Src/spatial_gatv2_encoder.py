"""
Spatial GATv2 Encoder for Multi-Graph Spatio-Temporal Forecasting.

This module implements a single shared GATv2 encoder whose parameters are
reused across the physical, wind, and semantic graphs. Each graph retains its
own learnable adjacency-driven structural bias, while the graph attention
weights themselves remain shared.

Input and output dimensionality:
- Single-graph encoder: expects node features of shape [N, F_in] and returns
  embeddings of shape [N, F_out].
- Batched encoder: expects node features of shape [B, N, F_in] and returns
  embeddings of shape [B, N, F_out].
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax


class StructuralBiasGATv2(MessagePassing):
    """
    GATv2 with an adjacency-induced structural bias.

    The same attention parameters are reused across graphs, while a graph-specific
    scalar beta_g adjusts the log-adjacency contribution for that graph.
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 4,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
        share_weights: bool = True,
    ):
        super().__init__(aggr="add", node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = Linear(
                in_channels,
                heads * out_channels,
                bias=False,
                weight_initializer="glorot",
            )
            self.lin_r = self.lin_l if share_weights else Linear(
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
            self.lin_r = self.lin_l if share_weights else Linear(
                in_channels[1],
                heads * out_channels,
                bias=False,
                weight_initializer="glorot",
            )

        self.att = Parameter(torch.empty(1, heads, out_channels))

        total_out_channels = heads * out_channels if concat else out_channels
        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.lin_r is not self.lin_l:
            self.lin_r.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_weight: Optional[Tensor] = None,
        graph_beta: Optional[Tensor] = None,
    ):
        H = self.heads
        C = self.out_channels

        if isinstance(x, Tensor):
            x_l = self.lin_l(x).view(-1, H, C)
            x_r = x_l if self.share_weights else self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x
            x_l = self.lin_l(x_l).view(-1, H, C)
            x_r = self.lin_r(x_r).view(-1, H, C)

        alpha = self.edge_updater(
            edge_index,
            x=(x_l, x_r),
            edge_weight=edge_weight,
            graph_beta=graph_beta,
        )

        out = self.propagate(
            edge_index,
            x=(x_l, x_r),
            alpha=alpha,
        )

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
        graph_beta: Optional[Tensor] = None,
    ) -> Tensor:
        x = x_i + x_j
        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        logits = (x * self.att).sum(dim=-1)

        if edge_weight is not None:
            edge_weight = edge_weight.clamp_min(1e-8)
            log_edge_weight = torch.log(edge_weight)

            if dim_size is None:
                dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0

            counts = torch.zeros(dim_size, device=log_edge_weight.device, dtype=log_edge_weight.dtype)
            counts.scatter_add_(0, index, torch.ones_like(log_edge_weight))

            mu = torch.zeros(dim_size, device=log_edge_weight.device, dtype=log_edge_weight.dtype)
            mu.scatter_add_(0, index, log_edge_weight)
            mu = mu / counts.clamp_min(1)

            if graph_beta is None:
                graph_beta = 1.0

            graph_beta = torch.as_tensor(graph_beta, device=log_edge_weight.device, dtype=log_edge_weight.dtype)
            bias_term = graph_beta * (log_edge_weight - mu[index])
            logits = logits + bias_term.unsqueeze(-1)

        alpha = softmax(logits, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
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
    Multi-graph spatial encoder that reuses one GATv2 layer across all graphs.
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
        self.gat = StructuralBiasGATv2(
            in_features,
            out_features,
            heads=heads,
            concat=True,
            negative_slope=0.2,
            dropout=dropout,
            bias=True,
            share_weights=True,
        )
        self.graph_betas = nn.Parameter(torch.ones(n_graphs))

    def forward(
        self,
        x: torch.Tensor,
        edge_indices: list,
        edge_weights: list,
    ) -> list:
        embeddings = []
        for graph_idx, (edge_index, edge_weight) in enumerate(zip(edge_indices, edge_weights)):
            emb = self.gat(
                x,
                edge_index,
                edge_weight=edge_weight,
                graph_beta=self.graph_betas[graph_idx],
            )
            embeddings.append(emb)
        return embeddings


class SpatialGATv2EncoderBatched(nn.Module):
    """
    Batched multi-graph encoder that reuses the same GATv2 layer across graphs.

    The graph identity is retained by using a separate beta_g for each adjacency,
    while the shared attention parameters remain common to all graphs.
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
        self.gat = StructuralBiasGATv2(
            in_features,
            out_features,
            heads=heads,
            concat=True,
            negative_slope=0.2,
            dropout=dropout,
            bias=True,
            share_weights=True,
        )
        self.graph_betas = nn.Parameter(torch.ones(n_graphs))

    @staticmethod
    def _flatten_batched_dense_adjacencies(
        edge_index: Tensor,
        edge_weight: Tensor,
        num_nodes: int,
    ) -> Tuple[Tensor, Tensor]:
        if edge_weight.dim() == 2:
            return edge_index, edge_weight

        if edge_weight.dim() != 3:
            raise ValueError(f"Expected dense adjacency with shape [B, N, N], got {tuple(edge_weight.shape)}")

        edge_indices = []
        edge_weights = []
        batch_size = edge_weight.size(0)

        for batch_idx in range(batch_size):
            A_b = edge_weight[batch_idx]
            edges = torch.nonzero(A_b, as_tuple=False)
            if edges.numel() == 0:
                continue
            src = edges[:, 0]
            dst = edges[:, 1]
            weights = A_b[src, dst]
            src = src + batch_idx * num_nodes
            dst = dst + batch_idx * num_nodes
            edge_indices.append(torch.stack([src, dst], dim=0))
            edge_weights.append(weights)

        if not edge_indices:
            return torch.empty((2, 0), device=edge_weight.device, dtype=torch.long), torch.empty((0,), device=edge_weight.device, dtype=edge_weight.dtype)

        return torch.cat(edge_indices, dim=1), torch.cat(edge_weights, dim=0)

    def forward(
        self,
        x_batch: torch.Tensor,
        edge_indices: list,
        edge_weights: list,
    ) -> list:
        B, N, F_in = x_batch.shape
        embeddings = []

        for graph_idx, (edge_index, edge_weight) in enumerate(zip(edge_indices, edge_weights)):
            x_flat = x_batch.reshape(B * N, F_in)

            if edge_weight is not None:
                if edge_weight.dim() == 3:
                    edge_index_flat, edge_weight_flat = self._flatten_batched_dense_adjacencies(
                        edge_index,
                        edge_weight,
                        N,
                    )
                else:
                    edge_index_flat, edge_weight_flat = edge_index, edge_weight
            else:
                edge_index_flat, edge_weight_flat = edge_index, None

            emb_flat = self.gat(
                x_flat,
                edge_index_flat,
                edge_weight=edge_weight_flat,
                graph_beta=self.graph_betas[graph_idx],
            )
            emb_batch = emb_flat.reshape(B, N, -1)
            embeddings.append(emb_batch)

        return embeddings
