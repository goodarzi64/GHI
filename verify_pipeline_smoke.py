import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, 'c:/Users/Mohsen/Documents/GHI')
from Src.data_structure import load_windowed_drive

root = Path(tempfile.mkdtemp())
(root / 'precomputed_graphs').mkdir(parents=True, exist_ok=True)

np.savez(
    root / 'filtered_tensor.npz',
    tensor=np.random.randn(20, 6, 3).astype(np.float32),
    mask_forecast=np.array([True, False, True]),
)
adj = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
np.savez(root / 'precomputed_graphs' / 'static_graph.npz', static_graph=adj)

dataset = load_windowed_drive(
    str(root),
    T=3,
    H=1,
    stride=1,
    val_duration=0,
    test_duration=3,
    tensor_array_key='tensor',
    input_mask_key='mask_forecast',
    output_mask_key='mask_forecast',
    graph_array_keys={'static_graph': 'static_graph'},
)
print('train_x_shape', dataset.train.x.shape)
print('graph_keys', list(dataset.train.historical_graphs.keys()))
