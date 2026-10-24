# GHI forecasting

Utilities and PyTorch modules for preparing windowed global horizontal irradiance
(GHI) data, encoding spatial and temporal features, evolving latent states across
forecast horizons, and predicting future spatial graphs and GHI.

## Repository layout

- `Src/data_structure.py`: chronological train/validation/test split, sliding windows,
  feature masks, and loading tensor and graph arrays from `.npz` files. Optional raw
  targets are exposed as `y_raw` for evaluation.
- `Src/Graph_build.py`: geographic, DTW, and wind based graph construction.
- `Src/MSRL.py` and `Src/fusion_module.py`: shared GATv2 spatial encoding with
  graph-specific structural bias and node-adaptive multi-graph fusion. The pipeline
  folds batch and time together so spatial encoding processes all time positions
  in one pass (or in optional chunks).
- `Src/MTRL.py`: causal temporal context encoder. It accepts `[B, T, N, C]` and
  returns `[B, W, N, 2*C_out]`, downsampling the temporal axis by two in each
  block while enough time steps remain, then applying a per-position channel
  projection from `C_out` to `2*C_out`.
- `Src/LGEE.py`: horizon-aware latent context generation, mapping
  `[B, W, N, C]` to `[B, H, N, C]`.
- `Src/FSDG.py` and `Src/FSDP.py`: future spatial dependency generation and
  multi-graph propagation.
- `Src/forecast_head.py`: horizon-wise node forecasts and forecast/graph loss.
- `tests/`: focused module and pipeline tests.

## Install

Install the packages listed in `requirements.txt` and PyTorch. `Src/MSRL.py`
also requires PyTorch Geometric; install a build compatible with your PyTorch
version using the [PyG installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Load windowed data

`load_windowed_drive` expects a directory containing
`filtered_tensor.npz` and a `precomputed_graphs` directory. It requires explicit
array keys for the tensor and each graph artifact that exists. The defaults
look for `static_graph.npz`, `wind_cloud_graphs.npz`, and `dtw_graphs.npz`;
missing graph files are skipped. Each graph array may be static `[N, N]` or
time-varying `[T, N, N]`.

```python
from Src.data_structure import load_windowed_drive

dataset = load_windowed_drive(
    "/path/to/filtered_data",
    T=12,                         # history length
    H=3,                          # forecast horizon
    stride=1,
    val_duration=365,
    test_duration=365,
    tensor_array_key="tensor",
    input_mask_key="mask_forecast",
    output_mask_key="mask_target_norm",
    raw_output_mask="mask_target_raw",  # optional raw target for evaluation
    graph_array_keys={
        "static_graph": "A_topk",
        "wind_cloud_graphs": "A_topk",
        "dtw_graphs": "A_topk",
    },
)

print(dataset.train.x.shape)  # [B, T, N, F_in]
print(dataset.train.y.shape)  # [B, H, N, F_out], normalized training target
print(dataset.train.y_raw.shape)  # [B, H, N], raw target when requested
print(dataset.train.historical_graphs.keys())
```

Set `val_duration=0` to produce one train/test split; in that case
`dataset.val` is `None`. Graph mappings contain `[N, N]` arrays for static
graphs and `[B, T, N, N]` / `[B, H, N, N]` windows for time-varying historical
and future graphs.

The loader creates in-memory sliding windows. Windows that refer to the same
source array are views where possible. Split durations are measured in source
time steps and must leave enough time for the requested history and horizon.

In the GHI pipeline, `y` contains `GHI_norm` and is used directly by the training
loss. Optional `y_raw` contains raw GHI in W/m² and is used for validation and test
reporting. Predictions are converted with the training-set mean and standard
deviation saved in `filtered_tensor.npz`, then clamped at zero before comparison
with `y_raw`. The saved `feature_means` and `feature_stds` were fit on 2017–2018;
validation and test data do not contribute to these statistics. RMSE, MAE, and R²
are reported in W/m², overall and by horizon, for all hours and daytime.

## Temporal encoder example

```python
import torch
from Src.MTRL import TemporalContextEncoder

x = torch.randn(4, 12, 20, 64)  # [batch, history, nodes, channels]
encoder = TemporalContextEncoder(
    in_channels=64,
    out_channels=64,
    num_blocks=3,
    kernel_size=3,
    dilation_list=[1, 2, 4],
)
z_hist = encoder(x)             # [4, 1, 20, 128]
```

The temporal encoder processes each node independently. Its convolutions are
causal, and each block halves the temporal length with average pooling. The
final linear projection doubles the channel width independently for each node
and remaining time position. Configure downstream latent modules with
`2 * out_channels`; the pipeline notebook uses this width for LGEE, future graph
generation, graph propagation, and the forecast head.

In `SpatiotemporalForecastingPipeline`, `input_length` determines the expected MTRL
history length: `input_length // 2**num_blocks`. The input length must be divisible
by `2**num_blocks`, since each block downsamples by two. The pipeline defaults to
three blocks and dilations `[1, 2, 4]`; the notebook training setup uses four blocks
and `[1, 2, 4, 8]`.

## Spatial time folding

The pipeline reshapes spatial inputs from `[B, T, N, F]` to `[B*T, N, F]` and runs
MSRL and graph fusion over the folded batch. Dynamic graphs are reshaped in the same
way, so each `(batch, time)` item retains its own graph. Static `[N, N]` graphs are
shared across items. Set `spatial_chunk` to a positive integer to process the folded
axis in chunks when memory is limited; `None` uses one pass. The fused result is
reshaped to `[B, T, N, C]` before temporal modeling.

MSRL uses the sparse top-k adjacency for each graph (`A_topk`), including DTW.
The pipeline may also retain dense `A_dtw` for the separate semantic anchor path;
that dense matrix is not the DTW adjacency passed to the spatial GAT.

## Spatial dimensions in the pipeline notebook

`SpatialGATv2EncoderBatched` currently constructs its GATv2 layer with
`concat=False`, so the attention heads are averaged. Each graph embedding has
width `out_features`, regardless of the number of heads. In
`NoteBooks/pipeline.ipynb`, the model sets `spatial_dim = spatial_hidden_dim`
and passes that width to the fusion module and as `in_channels` to the temporal
encoder.

The fusion module also concatenates the graph embeddings temporarily to score
their relative importance. It then combines the embeddings with a weighted sum,
so the fused output remains the same width as each graph embedding. This graph
fusion concatenation is separate from head aggregation.
