from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


TimeSlice = Tuple[int, int]
MaskLike = Optional[str | Sequence[bool] | Sequence[int] | np.ndarray]


@dataclass(frozen=True)
class TemporalSplit:
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
    - ``x``: [B, T, N, F_in]
    - ``y``: [B, H, N, F_out]
    - ``historical_graphs[name]``: [N, N] for static graphs or [B, T, N, N]
    - ``future_graphs[name]``: [N, N] for static graphs or [B, H, N, N]
    - ``cov_future``: optional [B, H, N, K] deterministic future covariates
    """

    x: np.ndarray
    y: np.ndarray
    historical_graphs: Dict[str, np.ndarray]
    future_graphs: Dict[str, np.ndarray]
    cov_future: Optional[np.ndarray] = None
    y_raw: Optional[np.ndarray] = None


@dataclass(frozen=True)
class WindowedDataset:
    """Windowed train/validation/test data for one chronological split."""

    name: str
    train: WindowedSplit
    val: Optional[WindowedSplit]
    test: WindowedSplit
    slices: TemporalSplit


def make_temporal_split(
    n_timesteps: int,
    *,
    val_duration: int,
    test_duration: int,
) -> TemporalSplit:
    """
    Build one chronological train/validation/test split.

    Args:
        n_timesteps: Total temporal length ``T``.
        val_duration: Validation length. If ``0``, validation is omitted.
        test_duration: Required final test length.

    Returns:
        The test split is always the last ``test_duration`` timesteps.
    """
    _validate_positive("n_timesteps", n_timesteps)
    _validate_positive("test_duration", test_duration)

    if test_duration >= n_timesteps:
        raise ValueError("test_duration must be smaller than n_timesteps.")

    available = n_timesteps - test_duration
    test_slice = (available, n_timesteps)

    if val_duration < 0:
        raise ValueError("val_duration cannot be negative.")
    if val_duration == 0:
        _validate_split_lengths(available, 0, test_duration, n_timesteps)
        return TemporalSplit(
            name="train_test",
            train_slice=(0, available),
            val_slice=None,
            test_slice=test_slice,
        )

    train_duration = available - val_duration
    _validate_split_lengths(train_duration, val_duration, test_duration, n_timesteps)
    return TemporalSplit(
        name="train_validation_test",
        train_slice=(0, train_duration),
        val_slice=(train_duration, available),
        test_slice=test_slice,
    )


def build_windowed_dataset(
    tensor: np.ndarray,
    graphs: Mapping[str, np.ndarray],
    split: TemporalSplit,
    *,
    T: int,
    H: int,
    stride: int = 1,
    input_mask: MaskLike = None,
    output_mask: MaskLike = None,
    cov_mask: MaskLike = None,
    target_tensor: Optional[np.ndarray] = None,
    raw_target_tensor: Optional[np.ndarray] = None,
) -> WindowedDataset:
    """
    Convert temporal data and graph arrays into one windowed dataset.

    Args:
        tensor: Node features shaped [T, N, F].
        graphs: Graph arrays shaped [N, N] or [T, N, N].
        split: Split definition from ``make_temporal_split``.
        T: Historical input window length.
        H: Future horizon length.
        stride: Sliding window stride.
        input_mask: Feature mask/key used for ``x``.
        output_mask: Feature mask/key used for ``y`` when ``target_tensor`` is
            not provided.
        cov_mask: Optional boolean feature mask or ordered feature indices for
            deterministic future covariates.
        target_tensor: Optional target array shaped [T, N] or [T, N, F_out].
        raw_target_tensor: Optional raw-unit target array with the same time and
            node dimensions, windowed into ``y_raw`` for evaluation.
    """
    _validate_temporal_tensor(tensor, "tensor")
    _validate_positive("T", T)
    _validate_positive("H", H)
    _validate_positive("stride", stride)

    x_source = _select_features(tensor, input_mask)
    cov_source = _select_features(tensor, cov_mask) if cov_mask is not None else None
    if target_tensor is None:
        y_source = _select_features(tensor, output_mask)
    else:
        y_source = _ensure_target_rank(target_tensor)
    raw_y_source = (
        _ensure_target_rank(raw_target_tensor)
        if raw_target_tensor is not None
        else None
    )

    if y_source.shape[:2] != tensor.shape[:2]:
        raise ValueError("target_tensor must share [T, N] with tensor.")
    if raw_y_source is not None and raw_y_source.shape[:2] != tensor.shape[:2]:
        raise ValueError("raw_target_tensor must share [T, N] with tensor.")

    selected_graphs = _validate_graphs(graphs, tensor.shape[0], tensor.shape[1])
    train = build_windowed_split(
        x_source,
        y_source,
        selected_graphs,
        split.train_slice,
        T=T,
        H=H,
        stride=stride,
        cov_source=cov_source,
        raw_y_source=raw_y_source,
    )
    val = (
        build_windowed_split(
            x_source,
            y_source,
            selected_graphs,
            split.val_slice,
            T=T,
            H=H,
            stride=stride,
            cov_source=cov_source,
            raw_y_source=raw_y_source,
        )
        if split.val_slice is not None
        else None
    )
    test = build_windowed_split(
        x_source,
        y_source,
        selected_graphs,
        split.test_slice,
        T=T,
        H=H,
        stride=stride,
        cov_source=cov_source,
        raw_y_source=raw_y_source,
    )
    return WindowedDataset(name=split.name, train=train, val=val, test=test, slices=split)


def build_windowed_split(
    x_source: np.ndarray,
    y_source: np.ndarray,
    graphs: Mapping[str, np.ndarray],
    time_slice: TimeSlice,
    *,
    T: int,
    H: int,
    stride: int = 1,
    cov_source: Optional[np.ndarray] = None,
    raw_y_source: Optional[np.ndarray] = None,
) -> WindowedSplit:
    """Build one split with ``x``, ``y``, historical graphs, and future graphs."""
    start, end = time_slice
    last_start = end - T - H
    if last_start < start:
        raise ValueError(
            f"Slice {time_slice} is too short for T={T} and H={H}; "
            f"it needs at least {T + H} timesteps."
        )

    starts = np.arange(start, last_start + 1, stride, dtype=np.int64)
    # Keep overlapping feature and target windows as views as well; stacking
    # them would duplicate the same source timesteps thousands of times.
    x = _window_series(x_source, starts, T, offset=0)
    y = _window_series(y_source, starts, H, offset=T)
    cov_future = (
        _window_series(cov_source, starts, H, offset=T)
        if cov_source is not None
        else None
    )
    y_raw = (
        _window_series(raw_y_source, starts, H, offset=T)
        if raw_y_source is not None
        else None
    )
    if y_raw is not None and y_raw.shape[-1] == 1:
        y_raw = y_raw[..., 0]

    historical_graphs: Dict[str, np.ndarray] = {}
    future_graphs: Dict[str, np.ndarray] = {}
    for name, graph in graphs.items():
        historical_graphs[name] = _window_graph(graph, starts, T, offset=0)
        future_graphs[name] = _window_graph(graph, starts, H, offset=T)

    return WindowedSplit(
        x=x,
        y=y,
        historical_graphs=historical_graphs,
        future_graphs=future_graphs,
        cov_future=cov_future,
        y_raw=y_raw,
    )


def load_windowed_drive(
    base_dir: str,
    *,
    T: int,
    H: int,
    val_duration: int,
    test_duration: int,
    stride: int = 1,
    tensor_filename: str = "filtered_tensor.npz",
    tensor_array_key: str,
    input_mask_key: Optional[str] = "mask_forecast",
    output_mask_key: Optional[str] = None,
    raw_output_mask: MaskLike = None,
    cov_mask: MaskLike = None,
    graph_dirname: str = "precomputed_graphs",
    graph_filenames: Sequence[str] = (
        "static_graph.npz",
        "wind_cloud_graphs.npz",
        "dtw_graphs.npz",
    ),
    graph_array_keys: Mapping[str, str | Sequence[str]],
    allow_pickle: bool = True,
) -> WindowedDataset:
    """
    Load ``filtered_tensor.npz`` and selected graph files, then return
    one in-memory windowed dataset. ``tensor_array_key`` and an explicit key for
    each existing graph artifact must be provided.

    ``graph_array_keys`` maps each graph artifact to its array key. Mapping
    keys can be artifact names such as ``"static_graph"`` or filenames such
    as ``"static_graph.npz"``. Graph array inference is disabled. Missing
    graph files are skipped; each existing graph file must have a mapping entry.
    """
    if graph_array_keys is None:
        raise ValueError("graph_array_keys is required and cannot be None.")

    tensor_arrays = load_npz_arrays(
        os.path.join(base_dir, tensor_filename),
        allow_pickle=allow_pickle,
    )
    tensor = select_array(tensor_arrays, tensor_array_key, ndim=3)

    input_mask = resolve_mask(tensor_arrays, input_mask_key)
    output_mask = resolve_mask(tensor_arrays, output_mask_key)
    raw_target_mask = resolve_mask(tensor_arrays, raw_output_mask)
    cov_mask_resolved = resolve_mask(tensor_arrays, cov_mask)
    graphs = load_graph_arrays(
        os.path.join(base_dir, graph_dirname),
        graph_filenames=graph_filenames,
        graph_array_keys=graph_array_keys,
        allow_pickle=allow_pickle,
    )

    split = make_temporal_split(
        tensor.shape[0],
        val_duration=val_duration,
        test_duration=test_duration,
    )
    return build_windowed_dataset(
        tensor,
        graphs,
        split,
        T=T,
        H=H,
        stride=stride,
        input_mask=input_mask,
        output_mask=output_mask,
        raw_target_tensor=(
            _select_features(tensor, raw_target_mask)
            if raw_target_mask is not None
            else None
        ),
        cov_mask=cov_mask_resolved,
    )


def load_graph_arrays(
    graph_dir: str,
    *,
    graph_filenames: Sequence[str] = (
        "static_graph.npz",
        "wind_cloud_graphs.npz",
        "dtw_graphs.npz",
    ),
    graph_array_keys: Mapping[str, str | Sequence[str]],
    allow_pickle: bool = True,
) -> Dict[str, np.ndarray]:
    """Load explicitly keyed arrays from existing graph files.

    ``graph_array_keys`` must map each existing artifact name or filename to
    an array key. Graph array inference is intentionally disabled.
    """
    if graph_array_keys is None:
        raise ValueError("graph_array_keys is required and cannot be None.")
    graphs: Dict[str, np.ndarray] = {}

    for filename in graph_filenames:
        artifact_name = os.path.splitext(filename)[0]
        path = os.path.join(graph_dir, filename)
        if not os.path.exists(path):
            continue

        arrays = load_npz_arrays(path, allow_pickle=allow_pickle)
        array_keys = graph_array_keys.get(artifact_name, graph_array_keys.get(filename))
        if isinstance(array_keys, str):
            graphs[artifact_name] = select_array(arrays, array_keys, ndim=None)
        elif array_keys is not None:
            for array_key in array_keys:
                graphs[f"{artifact_name}:{array_key}"] = select_array(
                    arrays, array_key, ndim=None
                )

    if not graphs:
        raise FileNotFoundError(f"No graph files found in: {graph_dir}")
    return graphs


def select_array(
    arrays: Mapping[str, np.ndarray],
    array_key: Optional[str],
    ndim: int | None = None,
    *,
    name: str = "array",
) -> np.ndarray:
    """
    Select an array using an explicit key; automatic inference is disabled.
    """
    if not array_key:
        raise ValueError(
            f"{name}: an explicit array_key is required; "
            "automatic array inference is disabled."
        )

    if array_key not in arrays:
        available = ", ".join(arrays.keys())
        raise KeyError(
            f"{name}: array key {array_key!r} was not found. "
            f"Available keys: {available}"
        )

    array = np.asarray(arrays[array_key])

    if ndim is not None and array.ndim != ndim:
        raise ValueError(
            f"{name}: array {array_key!r} must have ndim={ndim}, "
            f"but got shape {array.shape}."
        )

    return array

def resolve_mask(arrays: Mapping[str, np.ndarray], mask: MaskLike) -> Optional[np.ndarray]:
    """
    Resolve a boolean feature mask from a direct mask, an NPZ key, or a nested
    object-dict stored in the NPZ.
    """
    if mask is None:
        return None
    if not isinstance(mask, str):
        return np.asarray(mask)

    if mask in arrays:
        return np.asarray(arrays[mask], dtype=bool)

    for value in arrays.values():
        if isinstance(value, np.ndarray) and value.shape == ():
            item = value.item()
            if isinstance(item, Mapping) and mask in item:
                return np.asarray(item[mask], dtype=bool)

    raise KeyError(f"Mask key not found: {mask}")


def load_npz_arrays(npz_path: str, *, allow_pickle: bool = True) -> Dict[str, np.ndarray]:
    """Load all arrays from a compressed ``.npz`` artifact into a plain dict."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=allow_pickle) as data:
        return {key: data[key] for key in data.files}


def _select_features(array: np.ndarray, mask: MaskLike) -> np.ndarray:
    mask_array = np.asarray(mask) if mask is not None else None
    if mask_array is None:
        return array
    if mask_array.ndim != 1:
        raise ValueError(f"Feature mask must be one-dimensional, got {mask_array.shape}.")
    if mask_array.dtype == bool:
        if mask_array.shape[0] != array.shape[-1]:
            raise ValueError(
                f"Boolean feature mask must have shape [{array.shape[-1]}], got {mask_array.shape}."
            )
    elif not np.issubdtype(mask_array.dtype, np.integer):
        raise TypeError("Feature mask must contain booleans or integer feature indices.")
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
        return graph
    # A copied window tensor scales as [windows, length, N, N] and can dwarf
    # the source graph arrays. Sliding windows are views, so the source graph
    # storage is shared while batches can still index the usual [B,L,N,N] shape.
    window_width = offset + length
    windows = np.lib.stride_tricks.sliding_window_view(
        graph, window_width, axis=0
    ).transpose(0, 3, 1, 2)
    step = int(starts[1] - starts[0]) if len(starts) > 1 else 1
    return windows[int(starts[0]) : int(starts[-1]) + 1 : step, offset : offset + length]


def _window_series(series: np.ndarray, starts: np.ndarray, length: int, *, offset: int) -> np.ndarray:
    """Return [B,L,...] overlapping temporal windows as a shared-memory view."""
    window_width = offset + length
    windows = np.lib.stride_tricks.sliding_window_view(
        series, window_width, axis=0
    )
    # sliding_window_view appends the time-window axis at the end.
    windows = np.moveaxis(windows, -1, 1)
    step = int(starts[1] - starts[0]) if len(starts) > 1 else 1
    return windows[int(starts[0]) : int(starts[-1]) + 1 : step, offset : offset + length]


def _validate_temporal_tensor(array: np.ndarray, name: str) -> None:
    if array.ndim != 3:
        raise ValueError(f"{name} must be shaped [T,N,F], got {array.shape}.")



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

