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
    - ``x``: [B, T, N, F_in]
    - ``y``: [B, H, N, F_out]
    - ``historical_graphs[name]``: [N, N] for static graphs or [B, T, N, N]
    - ``future_graphs[name]``: [N, N] for static graphs or [B, H, N, N]
    """

    x: np.ndarray
    y: np.ndarray
    historical_graphs: Dict[str, np.ndarray]
    future_graphs: Dict[str, np.ndarray]


@dataclass(frozen=True)
class WindowedFold:
    """Train/validation/test windowed data for one fold."""

    name: str
    train: WindowedSplit
    val: Optional[WindowedSplit]
    test: WindowedSplit
    slices: TemporalFold


def make_expanding_folds(
    n_timesteps: int,
    *,
    val_duration: int,
    test_duration: int,
    fold_count: int = 1,
) -> List[TemporalFold]:
    """
    Build expanding validation folds with a fixed final test period.

    Args:
        n_timesteps: Total temporal length ``T``.
        val_duration: Required validation length per fold. If ``0``, no
            validation folds are created and one train/test split is returned.
        test_duration: Required final test length.
        fold_count: Number of expanding validation folds.

    The initial training duration is inferred from the timesteps left after
    reserving the final test period and all validation folds.

    Returns:
        A list of ``TemporalFold`` objects. Test is always the last
        ``test_duration`` timesteps.
    """
    _validate_positive("n_timesteps", n_timesteps)
    _validate_positive("fold_count", fold_count)
    _validate_positive("test_duration", test_duration)

    if test_duration >= n_timesteps:
        raise ValueError("test_duration must be smaller than n_timesteps.")

    available = n_timesteps - test_duration
    test_slice = (available, n_timesteps)

    if val_duration < 0:
        raise ValueError("val_duration cannot be negative.")
    if val_duration == 0:
        _validate_split_lengths(available, 0, test_duration, n_timesteps)
        return [
            TemporalFold(
                name="train_test",
                train_slice=(0, available),
                val_slice=None,
                test_slice=test_slice,
            )
        ]

    required_val = val_duration * fold_count
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


def build_windowed_folds(
    tensor: np.ndarray,
    graphs: Mapping[str, np.ndarray],
    folds: Sequence[TemporalFold],
    *,
    T: int,
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
        T: Historical input window length.
        H: Future horizon length.
        stride: Sliding window stride.
        input_mask: Feature mask/key used for ``x``.
        output_mask: Feature mask/key used for ``y`` when ``target_tensor`` is
            not provided.
        target_tensor: Optional target array shaped [T, N] or [T, N, F_out].
    """
    _validate_temporal_tensor(tensor, "tensor")
    _validate_positive("T", T)
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
            T=T,
            H=H,
            stride=stride,
        )
        val = (
            build_windowed_split(
                x_source,
                y_source,
                selected_graphs,
                fold.val_slice,
                T=T,
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
            T=T,
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
    T: int,
    H: int,
    stride: int = 1,
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
    x = np.stack([x_source[i : i + T] for i in starts], axis=0)
    y = np.stack([y_source[i + T : i + T + H] for i in starts], axis=0)

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
    )


def load_windowed_drive_folds(
    base_dir: str,
    *,
    T: int,
    H: int,
    val_duration: int,
    test_duration: int,
    stride: int = 1,
    fold_count: int = 1,
    tensor_filename: str = "filtered_tensor.npz",
    tensor_array_key: str,
    input_mask_key: Optional[str] = "mask_forecast",
    output_mask_key: Optional[str] = None,
    graph_dirname: str = "precomputed_graphs",
    graph_filenames: Sequence[str] = (
        "static_graph.npz",
        "wind_cloud_graphs.npz",
        "dtw_graphs.npz",
    ),
    graph_array_keys: Mapping[str, str | Sequence[str]],
    allow_pickle: bool = True,
) -> List[WindowedFold]:
    """
    Load ``filtered_tensor.npz`` and selected graph files, then return
    in-memory windowed folds. ``tensor_array_key`` and an explicit key for
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
    graphs = load_graph_arrays(
        os.path.join(base_dir, graph_dirname),
        graph_filenames=graph_filenames,
        graph_array_keys=graph_array_keys,
        allow_pickle=allow_pickle,
    )

    folds = make_expanding_folds(
        tensor.shape[0],
        val_duration=val_duration,
        test_duration=test_duration,
        fold_count=fold_count,
    )
    return build_windowed_folds(
        tensor,
        graphs,
        folds,
        T=T,
        H=H,
        stride=stride,
        input_mask=input_mask,
        output_mask=output_mask,
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
        return np.asarray(mask, dtype=bool)

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
        return graph
    return np.stack([graph[i + offset : i + offset + length] for i in starts], axis=0)


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

