# GHI Spatial Encoder & Fusion Module

Multi-graph spatial encoder using **GATv2** (Graph Attention Networks v2) with learnable edge bias, combined with a **fusion module** to merge outputs from three graph types (static, dynamic, wind).

## Files

- **`spatial_gatv2_encoder.py`**: GATv2 encoders with edge weight bias (β·log(A))
  - `GATv2EdgeBiasConv`: Single GATv2 layer with edge bias
  - `SpatialGATv2Encoder`: Unbatched encoder for 3 graphs
  - `SpatialGATv2EncoderBatched`: Batched version for Colab training

- **`fusion_module.py`**: Fusion strategies for combining graph embeddings
  - `AttentionFusion`: Learns importance per graph
  - `ConcatFusion`: Concatenate + project
  - `LearnedWeightFusion`: Fixed learnable weights
  - `SpatialFusionModule`: Main module with strategy selection
  - `MultiScaleFusion`: Hierarchical fusion (advanced)

- **`temporal_conv_module.py`**: Temporal convolution module for diurnal and seasonal kernels after spatial fusion
  - `TemporalBranch`: Conv1D branch with dilation and layer norm
  - `TemporalConvModule`: Temporal model that operates on fused spatial features over time and aggregates temporal kernels into one unified output

- **`spatial_encoder_example.py`**: Complete integration example & Colab usage

- **`Graph_build.py`**: Adjacency matrix construction (from GHI_forecasting)
  - Static graph (geographic distance)
  - Semantic graph (DTW on GHI/TCC history)
  - Wind graph (spatiotemporal wind propagation)

## Quick Start in Colab

### 1. Install Dependencies

```python
!pip install torch torch-geometric torch-geometric-temporal pyproj scikit-learn tqdm
```

### 2. Upload Module Files

```python
# Upload spatial_gatv2_encoder.py, fusion_module.py, Graph_build.py to Colab
```

### 3. Use in Notebook

```python
import torch
from spatial_gatv2_encoder import SpatialGATv2EncoderBatched
from fusion_module import SpatialFusionModule
from Graph_build import build_static_adjacency, build_semantic_adjacency

# Initialize encoder (3 GATv2 layers, one per graph)
encoder = SpatialGATv2EncoderBatched(
    in_features=8,        # your node feature dimension
    out_features=64,      # spatial embedding dimension
    n_graphs=3,           # static, dynamic, wind
    heads=8,              # attention heads per GATv2
)

# Initialize fusion (combine 3 embeddings)
fusion = SpatialFusionModule(
    embedding_dim=64,
    n_graphs=3,
    fusion_method='attention',  # or 'concat', 'learned'
)

# Forward pass
x = torch.randn(4, 20, 8)  # batch_size=4, num_nodes=20, features=8
embeddings = encoder(x, edge_indices, edge_weights)  # List of 3 embeddings
spatial_repr = fusion(embeddings)  # [4, 20, 64]
```

## Architecture

### Spatial GATv2 Encoder

For each graph type (static, dynamic, wind):
```
Node features [N, F_in]
    ↓
GATv2EdgeBiasConv(edge_weight → β·log(A))
    ↓
Node embeddings [N, F_out]
```

Edge bias formula:
$$e_{ij}^{(\ell)} = \beta \cdot \log(A_{ij})$$
where $A_{ij}$ is the adjacency weight, preventing $\log(0)$ via clamping.

### Fusion Module (Attention-based)

```
[emb_static, emb_dynamic, emb_wind]  (3 × [N, F])
    ↓
Attention scoring per graph
    ↓
Softmax weights α ∈ [0,1]³
    ↓
Fused: z = α₀·emb_static + α₁·emb_dynamic + α₂·emb_wind
```

## Parameters

### SpatialGATv2EncoderBatched
- `in_features`: Node feature dimension (e.g., 8 for [GHI, TCC, wind_speed, ...])
- `out_features`: Spatial embedding dimension (typically 32-128)
- `n_graphs`: Number of graph types (always 3 for GHI)
- `heads`: GATv2 attention heads (4, 8, 16)
- `dropout`: Dropout rate (0.1-0.2)

### SpatialFusionModule
- `embedding_dim`: Must match encoder's `out_features`
- `n_graphs`: Always 3
- `fusion_method`: 'attention' (learns weights), 'concat' (concatenate+project), 'learned' (fixed scalars)
- `dropout`: Dropout in fusion network

## Integration with Encoder-Decoder

Typical pipeline:
```
Input [B, W, N, F] (batch, window, nodes, features)
    ↓
Spatial encoder per time step (GATv2 + fusion) → [B, W, N, D_spatial]
    ↓
Temporal fusion module (diurnal + seasonal kernels) → [B, N, D_temporal]
    ↓
Decoder / prediction head → [B, H, N] predictions
```

With the new temporal module, a combined flow is:
```
Input [B, W, N, F]
    ↓
Spatial encoder + graph branch fusion at each time slice
    ↓
TemporalConvModule aggregation on fused spatial features
    ↓
Final node-level prediction or decoder input
```

Note: The temporal module receives the fused spatial representation and does not
need the original per-graph branch embeddings once fusion is complete.

## Edge Weights from Graph_build.py

Three adjacency types returned:

1. **Static (geographic)**:
   ```python
   A_static = build_static_adjacency(df_geo=df_geo, k=5)['A_sym_norm']
   ```

2. **Semantic (temporal coherence)**:
   ```python
   A_semantic = build_semantic_adjacency(ghi=ghi_hist, tcc=tcc_hist, k=5)['A_sym_norm']
   ```

3. **Wind (propagation)**:
   ```python
   wind_kernel = WindAdjacency(D_ij, Theta_ij)
   A_wind = wind_kernel(wind_feats, sparse=False)  # [B,N,N]
   ```

Convert to edge format:
```python
from torch_sparse import dense_to_sparse
edge_idx, edge_wt = dense_to_sparse(A)
```

## Troubleshooting

**ImportError: No module named 'torch_geometric'**
- Run: `!pip install torch-geometric`

**CUDA out of memory**
- Reduce batch size, number of heads, or embedding dimension
- Use `device='cpu'` for debugging

**NaN in edge weights**
- Check for isolated nodes (degree = 0)
- Use `torch.clamp(edge_weight, min=1e-8)` before `log()`

**Fusion weights not learning**
- Check gradients flow to fusion module
- Increase learning rate or fusion dropout
- Verify embeddings are not all zero

## Citation

Spatial module based on:
- **GATv2**: [Graph Attention Networks v2](https://arxiv.org/abs/2105.14491)
- Edge bias concept for graph kernels

## License

Same as GHI project
