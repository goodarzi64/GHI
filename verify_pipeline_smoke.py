import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, 'c:/Users/Mohsen/Documents/GHI')
from Src.data_splits import load_windowed_drive_folds

root = Path(tempfile.mkdtemp())
(root / 'precomputed_graphs').mkdir(parents=True, exist_ok=True)

np.savez(
    root / 'filtered_tensor.npz',
    tensor=np.random.randn(20, 6, 3).astype(np.float32),
    mask_forecast=np.array([True, False, True]),
)
adj = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
np.savez(root / 'precomputed_graphs' / 'static_graph.npz', static_graph=adj)

folds = load_windowed_drive_folds(
    str(root),
    W=3,
    H=1,
    stride=1,
    train_duration=None,
    val_duration=0,
    test_duration=3,
    fold_count=1,
    input_mask_key='mask_forecast',
    output_mask_key='mask_forecast',
)
print('fold_count', len(folds))
print('train_x_shape', folds[0].train.x.shape)
print('graph_keys', list(folds[0].train.historical_graphs.keys()))
