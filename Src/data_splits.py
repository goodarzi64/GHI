from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


TimeSlice = Tuple[int, int]
MaskLike = Optional[str | Sequence[bool] | np.ndarray]


@dataclass(frozen=True)
class TemporalFold:
    """
    One temporal train/validation/test split.

    Slices use Python's half-open convention: [start, end).
    If ``val_slice`` is ``None``, the split is a simple train/test holdout.
    """

    name: str
    train_slice: TimeSlice
    val_slice: Optional[TimeSlice]
    test_slice: TimeSlice


@dataclass(frozen=True)
class WindowedSplit:
    """
    Windowed arrays for one split.

    Shapes:
    - ``x``: [B, W, N, F_in]
    - ``y``: [B, H, N, F_out]
    - ``historical_graphs[name]``: [B, W, N, N]
    - ``future_graphs[name]``: [B, H, N, N]
    """

    x: np.ndarray
    y: np.ndarray
    historical_graphs: Dict[str, np.ndarray]
    future_graphs: Dict[str, np.ndarray]
    start_indices: np.ndarray


@dataclass(frozen=True)
class WindowedFold:
    """Train/validation/test windowed data for one fold."""

    name: str
    train: WindowedSplit
    val: Optional[WindowedSplit]
    test: WindowedSplit
    slices: TemporalFold


@dataclass(frozen=True)
class SplitArrays:
    """Legacy container for raw sliced arrays."""

    train: Dict[str, np.ndarray]
    val: Dict[str, np.ndarray]
    test: Dict[str, np.ndarray]


def make_expanding_folds(
    n_timesteps: int,
    *,
    train_duration: Optional[int] = None,
    val_duration: Optional[int] = None,
    test_duration: Optional[int] = None,
    fold_count: int = 1,
) -> List[TemporalFold]:
    """
    Build expanding validation folds with a fixed final test period.

    Args:
        n_timesteps: Total temporal length ``T``.
        train_duration: Initial train length. If omitted, it is inferred from
            the remaining timesteps after validation and test lengths.
        val_duration: Validation length per fold. If ``0``, no validation folds
            are created and one train/test split is returned.
        test_duration: Final test length. If omitted, defaults to one third of
            the series.
        fold_count: Number of expanding validation folds.

    Returns:
        A list of ``TemporalFold`` objects. Test is always the last
        ``test_duration`` timesteps.
    """
    _validate_positive("n_timesteps", n_timesteps)
    _validate_positive("fold_count", fold_count)

    if test_duration is None:
        test_duration = max(1, n_timesteps // 3)
    _validate_positive("test_duration", test_duration)

    if test_duration >= n_timesteps:
        raise ValueError("test_duration must be smaller than n_timesteps.")

    available = n_timesteps - test_duration
    test_slice = (available, n_timesteps)

    if val_duration is None:
        val_duration = max(0, available // (fold_count + 1))
    if val_duration < 0:
        raise ValueError("val_duration cannot be negative.")

    if val_duration == 0:
        train_len = available if train_duration is None else train_duration
        _validate_split_lengths(train_len, 0, test_duration, n_timesteps)
        return [
            TemporalFold(
                name="train_test",
                train_slice=(0, train_len),
                val_slice=None,
                test_slice=test_slice,
            )
        ]

    required_val = val_duration * fold_count
    if train_duration is None:
        train_duration = available - required_val
    _validate_split_lengths(train_duration, required_val, test_duration, n_timesteps)

    folds: List[TemporalFold] = []
    for fold_idx in range(fold_count):
        train_end = train_duration + fold_idx * val_duration
        val_start = train_end
        val_end = val_start + val_duration
        folds.append(
            TemporalFold(
                name=f"fold_{fold_idx + 1}",
                train_slice=(0, train_end),
                val_slice=(val_start, val_end),
                test_slice=test_slice,
            )
        )

    return folds


def make_three_year_folds(n_timesteps: int) -> List[TemporalFold]:
    """
    Backward-compatible 3-year split.

    Test is the final year. The second year is split into two expanding
    validation folds.
    """
    year_len = _resolve_equal_year_length(n_timesteps)
    return make_expanding_folds(
        n_timesteps,
        train_duration=year_len,
        val_duration=year_len // 2,
        test_duration=year_len,
        fold_count=2,
    )


def build_windowed_folds(
    tensor: np.ndarray,
    graphs: Mapping[str, np.ndarray],
    folds: Sequence[TemporalFold],
    *,
    W: int,
    H: int,
    stride: int = 1,
    input_mask: MaskLike = None,
    output_mask: MaskLike = None,
    target_tensor: Optional[np.ndarray] = None,
) -> List[WindowedFold]:
    """
    Convert temporal data and graph arrays into in-memory windowed folds.

    Args:
        tensor: Node features shaped [T, N, F].
        graphs: Graph arrays shaped [N, N] or [T, N, N].
        folds: Split definitions from ``make_expanding_folds``.
        W: Historical input window length.
        H: Future horizon length.
        stride: Sliding window stride.
        input_mask: Feature mask/key used for ``x``.
        output_mask: Feature mask/key used for ``y`` when ``target_tensor`` is
            not provided.
        target_tensor: Optional target array shaped [T, N] or [T, N, F_out].
    """
    _validate_temporal_tensor(tensor, "tensor")
    _validate_positive("W", W)
    _validate_positive("H", H)
    _validate_positive("stride", stride)

    x_source = _select_features(tensor, input_mask)
    if target_tensor is None:
        y_source = _select_features(tensor, output_mask)
    else:
        y_source = _ensure_target_rank(target_tensor)

    if y_source.shape[:2] != tensor.shape[:2]:
        raise ValueError("target_tensor must share [T, N] with tensor.")

    selected_graphs = _validate_graphs(graphs, tensor.shape[0], tensor.shape[1])
    windowed: List[WindowedFold] = []
    for fold in folds:
        train = build_windowed_split(
            x_source,
            y_source,
            selected_graphs,
            fold.train_slice,
            W=W,
            H=H,
            stride=stride,
        )
        val = (
            build_windowed_split(
                x_source,
                y_source,
                selected_graphs,
                fold.val_slice,
                W=W,
                H=H,
                stride=stride,
            )
            if fold.val_slice is not None
            else None
        )
        test = build_windowed_split(
            x_source,
            y_source,
            selected_graphs,
            fold.test_slice,
            W=W,
            H=H,
            stride=stride,
        )
        windowed.append(
            WindowedFold(
                name=fold.name,
                train=train,
                val=val,
                test=test,
                slices=fold,
            )
        )

    return windowed


def build_windowed_split(
    x_source: np.ndarray,
    y_source: np.ndarray,
    graphs: Mapping[str, np.ndarray],
    time_slice: TimeSlice,
    *,
    W: int,
    H: int,
    stride: int = 1,
) -> WindowedSplit:
    """Build one split with ``x``, ``y``, historical graphs, and future graphs."""
    start, end = time_slice
    last_start = end - W - H
    if last_start < start:
        raise ValueError(
            f"Slice {time_slice} is too short for W={W} and H={H}; "
            f"it needs at least {W + H} timesteps."
        )

    starts = np.arange(start, last_start + 1, stride, dtype=np.int64)
    x = np.stack([x_source[i : i + W] for i in starts], axis=0)
    y = np.stack([y_source[i + W : i + W + H] for i in starts], axis=0)

    historical_graphs: Dict[str, np.ndarray] = {}
    future_graphs: Dict[str, np.ndarray] = {}
    for name, graph in graphs.items():
        historical_graphs[name] = _window_graph(graph, starts, W, offset=0)
        future_graphs[name] = _window_graph(graph, starts, H, offset=W)

    return WindowedSplit(
        x=x,
        y=y,
        historical_graphs=historical_graphs,
        future_graphs=future_graphs,
        start_indices=starts,
    )


def load_windowed_drive_folds(
    base_dir: str,
    *,
    W: int,
    H: int,
    stride: int = 1,
    train_duration: Optional[int] = None,
    val_duration: Optional[int] = None,
    test_duration: Optional[int] = None,
    fold_count: int = 1,
    tensor_filename: str = "filtered_tensor.npz",
    tensor_array_key: Optional[str] = None,
    input_mask_key: Optional[str] = "mask_forecast",
    output_mask_key: Optional[str] = None,
    target_array_key: Optional[str] = None,
    graph_dirname: str = "precomputed_graphs",
    graph_filenames: Sequence[str] = (
        "static_graph.npz",
        "wind_cloud_graphs.npz",
        "dtw_graphs.npz",
    ),
    graph_array_keys: Optional[Mapping[str, str]] = None,
    allow_pickle: bool = True,
) -> List[WindowedFold]:
    """
    Load ``filtered_tensor.npz`` and all selected graph files, then return
    in-memory windowed folds.

    ``graph_array_keys`` may select one array per graph artifact. Keys can be
    artifact names such as ``"static_graph"`` or filenames such as
    ``"static_graph.npz"``. If omitted, the first graph-like array is used.
    """
    tensor_arrays = load_npz_arrays(
        os.path.join(base_dir, tensor_filename),
        allow_pickle=allow_pickle,
    )
    tensor = select_array(tensor_arrays, tensor_array_key, ndim=3)

    input_mask = resolve_mask(tensor_arrays, input_mask_key)
    output_mask = resolve_mask(tensor_arrays, output_mask_key)
    target_tensor = (
        select_array(tensor_arrays, target_array_key, ndim=None)
        if target_array_key is not None
        else None
    )

    graphs = load_graph_arrays(
        os.path.join(base_dir, graph_dirname),
        graph_filenames=graph_filenames,
        graph_array_keys=graph_array_keys,
        allow_pickle=allow_pickle,
    )

    folds = make_expanding_folds(
        tensor.shape[0],
        train_duration=train_duration,
        val_duration=val_duration,
        test_duration=test_duration,
        fold_count=fold_count,
    )
    return build_windowed_folds(
        tensor,
        graphs,
        folds,
        W=W,
        H=H,
        stride=stride,
        input_mask=input_mask,
        output_mask=output_mask,
        target_tensor=target_tensor,
    )


def load_graph_arrays(
    graph_dir: str,
    *,
    graph_filenames: Sequence[str] = (
        "static_graph.npz",
        "wind_cloud_graphs.npz",
        "dtw_graphs.npz",
    ),
    graph_array_keys: Optional[Mapping[str, str]] = None,
    allow_pickle: bool = True,
) -> Dict[str, np.ndarray]:
    """Load one selected array from each graph file."""
    graph_array_keys = graph_array_keys or {}
    graphs: Dict[str, np.ndarray] = {}

    for filename in graph_filenames:
        artifact_name = os.path.splitext(filename)[0]
        path = os.path.join(graph_dir, filename)
        if not os.path.exists(path):
            continue

        arrays = load_npz_arrays(path, allow_pickle=allow_pickle)
        array_key = graph_array_keys.get(artifact_name, graph_array_keys.get(filename))
        graphs[artifact_name] = select_array(arrays, array_key, ndim=None, graph_like=True)

    if not graphs:
        raise FileNotFoundError(f"No graph files found in: {graph_dir}")
    return graphs


def select_array(
    arrays: Mapping[str, np.ndarray],
    array_key: Optional[str],
    *,
    ndim: Optional[int],
    graph_like: bool = False,
) -> np.ndarray:
    """Select one array by key, or infer a suitable array from an NPZ mapping."""
    if array_key is not None:
        if array_key not in arrays:
            raise KeyError(f"Array key not found: {array_key}")
        array = arrays[array_key]
        _validate_selected_array(array, array_key, ndim=ndim, graph_like=graph_like)
        return array

    for key, array in arrays.items():
        try:
            _validate_selected_array(array, key, ndim=ndim, graph_like=graph_like)
        except ValueError:
            continue
        return array

    expected = "graph-like [N,N] or [T,N,N]" if graph_like else f"{ndim}D"
    raise ValueError(f"Could not infer a {expected} array from keys: {list(arrays)}")


def resolve_mask(arrays: Mapping[str, np.ndarray], mask: MaskLike) -> Optional[np.ndarray]:
    """
    Resolve a boolean feature mask from a direct mask, an NPZ key, or a nested
    object-dict stored in the NPZ.
    """
    if mask is None:
        return None
    if not isinstance(mask, str):
        return np.asarray(mask, dtype=bool)

    if mask in arrays:
        return np.asarray(arrays[mask], dtype=bool)

    for value in arrays.values():
        if isinstance(value, np.ndarray) and value.shape == ():
            item = value.item()
            if isinstance(item, Mapping) and mask in item:
                return np.asarray(item[mask], dtype=bool)

    raise KeyError(f"Mask key not found: {mask}")


def split_arrays_by_fold(
    arrays: Mapping[str, np.ndarray],
    fold: TemporalFold,
    *,
    time_axis: int = 0,
    static_keys: Iterable[str] = (),
) -> SplitArrays:
    """Legacy raw array splitter."""
    static_key_set = set(static_keys)
    train: Dict[str, np.ndarray] = {}
    val: Dict[str, np.ndarray] = {}
    test: Dict[str, np.ndarray] = {}

    for key, array in arrays.items():
        if key in static_key_set:
            train[key] = array
            if fold.val_slice is not None:
                val[key] = array
            test[key] = array
            continue

        train[key] = slice_time(array, fold.train_slice, axis=time_axis)
        if fold.val_slice is not None:
            val[key] = slice_time(array, fold.val_slice, axis=time_axis)
        test[key] = slice_time(array, fold.test_slice, axis=time_axis)

    return SplitArrays(train=train, val=val, test=test)


def split_npz_by_fold(
    npz_path: str,
    fold: TemporalFold,
    *,
    time_axis: int = 0,
    static_keys: Iterable[str] = (),
    allow_pickle: bool = True,
) -> SplitArrays:
    """Load one ``.npz`` file and split all arrays by the selected fold."""
    arrays = load_npz_arrays(npz_path, allow_pickle=allow_pickle)
    return split_arrays_by_fold(
        arrays,
        fold,
        time_axis=time_axis,
        static_keys=static_keys,
    )


def load_npz_arrays(npz_path: str, *, allow_pickle: bool = True) -> Dict[str, np.ndarray]:
    """Load all arrays from a compressed ``.npz`` artifact into a plain dict."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=allow_pickle) as data:
        return {key: data[key] for key in data.files}


def load_drive_artifacts(
    base_dir: str,
    *,
    tensor_filename: str = "filtered_tensor.npz",
    graph_dirname: str = "precomputed_graphs",
    graph_filenames: Sequence[str] = (
        "static_graph.npz",
        "wind_cloud_graphs.npz",
        "dtw_graphs.npz",
    ),
    allow_pickle: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Legacy loader for raw tensor and graph artifacts."""
    artifacts: Dict[str, Dict[str, np.ndarray]] = {
        "tensor": load_npz_arrays(
            os.path.join(base_dir, tensor_filename),
            allow_pickle=allow_pickle,
        )
    }

    graph_dir = os.path.join(base_dir, graph_dirname)
    for filename in graph_filenames:
        key = os.path.splitext(filename)[0]
        path = os.path.join(graph_dir, filename)
        if os.path.exists(path):
            artifacts[key] = load_npz_arrays(path, allow_pickle=allow_pickle)

    return artifacts


def split_drive_artifacts(
    artifacts: Mapping[str, Mapping[str, np.ndarray]],
    fold: TemporalFold,
    *,
    time_axis: int = 0,
    static_artifact_names: Iterable[str] = ("static_graph",),
) -> Dict[str, SplitArrays]:
    """Legacy raw artifact splitter."""
    static_names = set(static_artifact_names)
    split_artifacts: Dict[str, SplitArrays] = {}

    for artifact_name, arrays in artifacts.items():
        is_static = artifact_name in static_names
        split_artifacts[artifact_name] = split_arrays_by_fold(
            arrays,
            fold,
            time_axis=time_axis,
            static_keys=arrays.keys() if is_static else (),
        )

    return split_artifacts


def describe_folds(folds: Sequence[TemporalFold]) -> List[Dict[str, int | str | None]]:
    """Return split boundaries and row counts for logging."""
    rows: List[Dict[str, int | str | None]] = []
    for fold in folds:
        val_start = fold.val_slice[0] if fold.val_slice is not None else None
        val_end = fold.val_slice[1] if fold.val_slice is not None else None
        val_rows = (
            fold.val_slice[1] - fold.val_slice[0]
            if fold.val_slice is not None
            else 0
        )
        rows.append(
            {
                "fold": fold.name,
                "train_start": fold.train_slice[0],
                "train_end": fold.train_slice[1],
                "train_rows": fold.train_slice[1] - fold.train_slice[0],
                "val_start": val_start,
                "val_end": val_end,
                "val_rows": val_rows,
                "test_start": fold.test_slice[0],
                "test_end": fold.test_slice[1],
                "test_rows": fold.test_slice[1] - fold.test_slice[0],
            }
        )
    return rows


def infer_n_timesteps(
    arrays: Mapping[str, np.ndarray],
    *,
    preferred_keys: Sequence[str] = ("tensor", "data", "A_sym_norm", "A_wind"),
    time_axis: int = 0,
) -> int:
    """Infer ``T`` from the first matching array in an artifact dictionary."""
    for key in preferred_keys:
        if key in arrays:
            return int(arrays[key].shape[time_axis])

    for array in arrays.values():
        if array.ndim > time_axis:
            return int(array.shape[time_axis])

    raise ValueError("Could not infer timesteps from empty or scalar arrays.")


def slice_time(array: np.ndarray, time_slice: TimeSlice, *, axis: int = 0) -> np.ndarray:
    """Slice a numpy array by time along any axis."""
    index = [slice(None)] * array.ndim
    index[axis] = slice(time_slice[0], time_slice[1])
    return array[tuple(index)]


def _select_features(array: np.ndarray, mask: MaskLike) -> np.ndarray:
    mask_array = np.asarray(mask, dtype=bool) if mask is not None else None
    if mask_array is None:
        return array
    if mask_array.ndim != 1 or mask_array.shape[0] != array.shape[-1]:
        raise ValueError(
            f"Feature mask must have shape [{array.shape[-1]}], got {mask_array.shape}."
        )
    return array[..., mask_array]


def _ensure_target_rank(target_tensor: np.ndarray) -> np.ndarray:
    if target_tensor.ndim == 2:
        return target_tensor[..., np.newaxis]
    if target_tensor.ndim == 3:
        return target_tensor
    raise ValueError("target_tensor must be shaped [T,N] or [T,N,F_out].")


def _validate_graphs(
    graphs: Mapping[str, np.ndarray],
    n_timesteps: int,
    n_nodes: int,
) -> Dict[str, np.ndarray]:
    selected: Dict[str, np.ndarray] = {}
    for name, graph in graphs.items():
        if graph.ndim == 2:
            if graph.shape != (n_nodes, n_nodes):
                raise ValueError(f"{name} must be shaped [N,N], got {graph.shape}.")
        elif graph.ndim == 3:
            if graph.shape != (n_timesteps, n_nodes, n_nodes):
                raise ValueError(f"{name} must be shaped [T,N,N], got {graph.shape}.")
        else:
            raise ValueError(f"{name} must be [N,N] or [T,N,N], got {graph.shape}.")
        selected[name] = graph
    return selected


def _window_graph(graph: np.ndarray, starts: np.ndarray, length: int, *, offset: int) -> np.ndarray:
    if graph.ndim == 2:
        return np.broadcast_to(graph, (len(starts), length, *graph.shape)).copy()
    return np.stack([graph[i + offset : i + offset + length] for i in starts], axis=0)


def _validate_temporal_tensor(array: np.ndarray, name: str) -> None:
    if array.ndim != 3:
        raise ValueError(f"{name} must be shaped [T,N,F], got {array.shape}.")


def _validate_selected_array(
    array: np.ndarray,
    key: str,
    *,
    ndim: Optional[int],
    graph_like: bool,
) -> None:
    if graph_like:
        if array.ndim not in (2, 3):
            raise ValueError(f"{key} is not graph-like.")
        if array.shape[-1] != array.shape[-2]:
            raise ValueError(f"{key} is not square on its last two axes.")
        return

    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{key} must be {ndim}D, got {array.ndim}D.")


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_split_lengths(
    train_duration: int,
    validation_total: int,
    test_duration: int,
    n_timesteps: int,
) -> None:
    _validate_positive("train_duration", train_duration)
    if train_duration + validation_total + test_duration > n_timesteps:
        raise ValueError(
            "train_duration + validation duration(s) + test_duration exceeds n_timesteps."
        )


def _resolve_equal_year_length(n_timesteps: int) -> int:
    """Validate the equal-year assumption and return one year length."""
    _validate_positive("n_timesteps", n_timesteps)
    if n_timesteps % 3 != 0:
        raise ValueError("n_timesteps must be divisible by 3 for equal 3-year data.")
    return n_timesteps // 3
