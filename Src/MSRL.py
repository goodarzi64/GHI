"""MSRL (Multi-graph Spatial Representation Learning).

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

            if graph_beta is None:
                graph_beta = 1.0

            graph_beta = torch.as_tensor(graph_beta, device=log_edge_weight.device, dtype=log_edge_weight.dtype)
            bias_term = graph_beta * log_edge_weight
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


class SpatialGATv2EncoderBatched(nn.Module):
    """
    Batched multi-graph encoder that reuses the same GATv2 layer across graphs.

    The graph identity is retained by using a separate beta_g for each adjacency,
    while the shared attention parameters remain common to all graphs. Dense
    adjacency matrices can be passed directly as ``[N, N]`` or ``[B, N, N]``
    tensors in ``adjacencies``. They use the convention ``A[dst, src]``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_graphs: int = 3,
        heads: int = 3,
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
            share_weights=False,
        )
        self.graph_betas = nn.Parameter(torch.ones(n_graphs))

    @staticmethod
    def _dense_adjacency_to_edges(
        adjacency: Tensor,
        batch_size: int,
        num_nodes: int,
    ) -> Tuple[Tensor, Tensor]:
        """Convert dense ``A[dst, src]`` matrices to PyG edges for a batch."""
        if adjacency.dim() == 2:
            if adjacency.shape != (num_nodes, num_nodes):
                raise ValueError(
                    f"Expected adjacency shape [{num_nodes}, {num_nodes}], "
                    f"got {tuple(adjacency.shape)}."
                )
            adjacency = adjacency.unsqueeze(0).expand(batch_size, -1, -1)
        elif adjacency.dim() == 3:
            if adjacency.shape != (batch_size, num_nodes, num_nodes):
                raise ValueError(
                    f"Expected adjacency shape [{batch_size}, {num_nodes}, {num_nodes}], "
                    f"got {tuple(adjacency.shape)}."
                )
        else:
            raise ValueError(
                f"Expected dense adjacency [N, N] or [B, N, N], got {tuple(adjacency.shape)}."
            )

        edge_indices = []
        edge_weights = []
        for batch_idx in range(batch_size):
            A_b = adjacency[batch_idx]
            edges = torch.nonzero(A_b, as_tuple=False)
            if edges.numel() == 0:
                continue
            dst = edges[:, 0]
            src = edges[:, 1]
            weights = A_b[dst, src]
            src = src + batch_idx * num_nodes
            dst = dst + batch_idx * num_nodes
            edge_indices.append(torch.stack([src, dst], dim=0))
            edge_weights.append(weights)

        if not edge_indices:
            return (
                torch.empty((2, 0), device=adjacency.device, dtype=torch.long),
                torch.empty((0,), device=adjacency.device, dtype=adjacency.dtype),
            )

        return torch.cat(edge_indices, dim=1), torch.cat(edge_weights, dim=0)

    def forward(
        self,
        x_batch: torch.Tensor,
        adjacencies: list,
        edge_weights: Optional[list] = None,
    ) -> list:
        """Encode each graph from dense adjacency matrices or legacy edge lists.

        Args:
            x_batch: Node features shaped ``[B, N, F_in]``.
            adjacencies: Dense matrices shaped ``[N, N]`` or ``[B, N, N]``
                when ``edge_weights`` is omitted. For backward compatibility,
                this may instead be a list of PyG ``edge_index`` tensors when
                ``edge_weights`` is also supplied.
            edge_weights: Optional list of edge weights paired with legacy
                ``edge_index`` inputs.
        """
        if x_batch.dim() != 3:
            raise ValueError(f"x_batch must have shape [B, N, F], got {tuple(x_batch.shape)}.")
        B, N, F_in = x_batch.shape
        if len(adjacencies) != self.n_graphs:
            raise ValueError(f"Expected {self.n_graphs} graphs, got {len(adjacencies)}.")
        if edge_weights is not None and len(edge_weights) != len(adjacencies):
            raise ValueError("edge_indices and edge_weights must contain the same number of graphs.")

        embeddings = []
        for graph_idx, graph in enumerate(adjacencies):
            x_flat = x_batch.reshape(B * N, F_in)

            if edge_weights is None:
                graph = graph.to(device=x_batch.device, dtype=x_batch.dtype)
                edge_index_flat, edge_weight_flat = self._dense_adjacency_to_edges(
                    graph, B, N
                )
            else:
                edge_index = graph
                edge_weight = edge_weights[graph_idx]
                if edge_weight is not None and edge_weight.dim() == 3:
                    edge_weight = edge_weight.to(device=x_batch.device, dtype=x_batch.dtype)
                    edge_index_flat, edge_weight_flat = self._dense_adjacency_to_edges(
                        edge_weight, B, N
                    )
                else:
                    edge_index_flat, edge_weight_flat = edge_index, edge_weight

            emb_flat = self.gat(
                x_flat,
                edge_index_flat,
                edge_weight=edge_weight_flat,
                graph_beta=self.graph_betas[graph_idx],
            )
            emb_batch = emb_flat.reshape(B, N, -1)
            embeddings.append(emb_batch)

        return embeddings
