# GHI
GHI forecasting 

## Windowed train/validation/test splits

Use `Src.data_splits` after mounting Google Drive in Colab. The helper loads
`filtered_tensor.npz` and the graph files under `precomputed_graphs`, applies
feature masks from the tensor file, and returns in-memory sliding windows:

- `x`: `[B, W, N, F_in]`
- `y`: `[B, H, N, F_out]`
- `historical_graphs[name]`: `[B, W, N, N]`
- `future_graphs[name]`: `[B, H, N, N]`

```python
from google.colab import drive

from Src.data_splits import (
    describe_folds,
    load_windowed_drive_folds,
)

drive.mount("/content/gdrive")

base_dir = "/content/gdrive/MyDrive/filtered_data"
folds = load_windowed_drive_folds(
    base_dir,
    W=12,
    H=3,
    stride=1,
    train_duration=None,
    val_duration=365,
    test_duration=365,
    fold_count=3,
    input_mask_key="mask_forecast",
    output_mask_key="mask_forecast",
    # Optional: graph_array_keys={"static_graph": "your_array_key"}
)

print(describe_folds([fold.slices for fold in folds]))
first_fold = folds[0]
print(first_fold.train.x.shape)
print(first_fold.train.y.shape)
print(first_fold.train.historical_graphs.keys())
```

Set `val_duration=0` to skip validation folds and return one train/test split.
