from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from .GST_Utils import topk_row
import torch.nn as nn

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    tqdm = None
#-------------------------------------------------------------------
#-----------------------static Graph building-----------------------
#-------------------------------------------------------------------

class GeoGeometry:
    """Compute pairwise WGS84 distances and bearings for geographic nodes.

    Parameters
    ----------
    df_geo
        Table with one row per node and ``latitude`` and ``longitude`` columns.
    device : str, default="cpu"
        Torch device on which to store the resulting matrices.

    Attributes
    ----------
    dist_matrix : torch.Tensor
        Pairwise geodesic distances in kilometers, with shape ``[N, N]``.
    theta_matrix : torch.Tensor
        Pairwise forward azimuths in radians, with shape ``[N, N]``.
    """

    def __init__(self, df_geo, device: str = "cpu") -> None:
        """Build and store distance and bearing matrices for ``df_geo``."""
        self.df_geo = df_geo
        self.device = device
        self._build()

    def _build(self) -> None:
        """Calculate WGS84 geodesics and assign the resulting tensors.

        Returns
        -------
        None
            Results are assigned to ``dist_matrix`` and ``theta_matrix``.
        """
        try:
            from pyproj import Geod
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pyproj is required for GeoGeometry. Install with `pip install pyproj`."
            ) from exc

        n = self.df_geo.shape[0]
        geod = Geod(ellps="WGS84")

        dist = np.zeros((n, n), dtype=np.float32)
        theta = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            lat1, lon1 = self.df_geo.iloc[i]["latitude"], self.df_geo.iloc[i]["longitude"]
            for j in range(n):
                if i == j:
                    continue
                lat2, lon2 = self.df_geo.iloc[j]["latitude"], self.df_geo.iloc[j]["longitude"]
                az12, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)
                dist[i, j] = dist_m / 1000.0
                theta[i, j] = np.radians(az12)

        self.dist_matrix = torch.tensor(dist, dtype=torch.float32, device=self.device)
        self.theta_matrix = torch.tensor(theta, dtype=torch.float32, device=self.device)


class DistanceKernel:
    """Convert pairwise distances into Gaussian-kernel edge weights.

    Parameters
    ----------
    dist_matrix : torch.Tensor
        Pairwise distances with shape ``[N, N]``.
    sigma : torch.Tensor or float or None, default=None
        Kernel width. If omitted or false-valued, it is estimated from the
        strict upper triangle of ``dist_matrix``.
    """

    def __init__(self, dist_matrix: torch.Tensor, sigma: torch.Tensor | float | None = None) -> None:
        """Initialize the distance matrix and select a kernel width."""
        self.dist_matrix = dist_matrix
        self.sigma = sigma or self._estimate_sigma()

    def _estimate_sigma(self) -> torch.Tensor:
        """Estimate kernel width from unique off-diagonal distances.

        Returns
        -------
        torch.Tensor
            Standard deviation of the strict upper-triangular distances.
        """
        mask = torch.triu(torch.ones_like(self.dist_matrix), diagonal=1).bool()
        return torch.std(self.dist_matrix[mask])

    def compute(self, self_loops: bool = False) -> torch.Tensor:
        """Compute Gaussian weights and optionally retain diagonal entries.

        Parameters
        ----------
        self_loops : bool, default=False
            If false, set the output diagonal to zero. If true, leave the
            Gaussian-kernel diagonal values unchanged.

        Returns
        -------
        torch.Tensor
            Dense weight matrix with shape ``[N, N]``.
        """
        A = torch.exp(- (self.dist_matrix ** 2) / (2 * self.sigma ** 2))
        if not self_loops:
            A.fill_diagonal_(0)
        return A


def build_geo_matrices(df_geo, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Build reusable geographic distance and bearing matrices.

    Parameters
    ----------
    df_geo
        Table with ``latitude`` and ``longitude`` columns, one row per node.
    device : str, default="cpu"
        Torch device for the returned tensors.

    Returns
    -------
    Dict[str, torch.Tensor]
        Mapping with ``dist_matrix`` (kilometers) and ``theta_matrix``
        (radians), each shaped ``[N, N]``.
    """
    geo = GeoGeometry(df_geo, device=device)
    return {
        "dist_matrix": geo.dist_matrix,
        "theta_matrix": geo.theta_matrix,
    }

def build_static_adjacency(
    dist_matrix: torch.Tensor,
    k: int = 5,
    self_loops: bool = False,
    topk_sym: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Build Gaussian-kernel static adjacency variants from geographic distances.

    Parameters
    ----------
    dist_matrix : torch.Tensor
        Pairwise geographic distances in kilometers with shape ``[N, N]``,
        typically returned by :func:`build_geo_matrices`.
    k : int, default=5
        Number of neighbors retained per row in ``A_topk``.
    self_loops : bool, default=False
        Whether the raw and top-k adjacency retain diagonal weights.
    topk_sym : bool, default=False
        Whether top-k sparsification is symmetrized.

    Returns
    -------
    Dict[str, torch.Tensor]
        Contains ``A_stat`` and ``A_topk``, each with shape ``[N, N]``.
    """
    if dist_matrix.ndim != 2 or dist_matrix.shape[0] != dist_matrix.shape[1]:
        raise ValueError("`dist_matrix` must be a square [N, N] tensor.")

    kernel = DistanceKernel(dist_matrix, sigma=None)
    A_stat = kernel.compute(self_loops=self_loops)

    out = {
        "A_stat": A_stat,
        "A_topk": topk_row(A_stat, k=k, sym=topk_sym, eps=1e-8, preserve_diagonal=self_loops),
    }
    return out

#-------------------------------------------------------------------
#-----------------------semantic Graph building---------------------
#-------------------------------------------------------------------
def dtw_distance(
    x: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute DTW accumulated cost, optimal warping-path length,
    and length-normalized exponential DTW similarity.

    Parameters
    ----------
    x : torch.Tensor
        First time series with shape [W1, F].

    y : torch.Tensor
        Second time series with shape [W2, F].

    Returns
    -------
    dtw_cost : torch.Tensor
        Accumulated DTW cost along the optimal warping path.

    path_length : torch.Tensor
        Number of aligned pairs along the optimal DTW path.

    similarity : torch.Tensor
        Exponential similarity based on the mean DTW alignment cost:

            similarity = exp(-dtw_cost / path_length)

        Therefore, the similarity is bounded in (0, 1], with
        similarity = 1 for identical sequences.
    """
    # ------------------------------------------------------------
    # 1. Validate input
    # ------------------------------------------------------------
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(
            "x and y must have shape [W, F]."
        )

    if x.shape[1] != y.shape[1]:
        raise ValueError(
            "x and y must have the same feature dimension."
        )

    # ------------------------------------------------------------
    # 2. Convert to floating point
    # ------------------------------------------------------------
    x = x.float()
    y = y.float()

    # ------------------------------------------------------------
    # 3. Local pairwise Euclidean distances
    #
    # D[i, j] = ||x[i] - y[j]||_2
    #
    # Shape: [W1, W2]
    # ------------------------------------------------------------
    D = torch.cdist(x, y, p=2)

    W1, W2 = D.shape

    # ------------------------------------------------------------
    # 4. Accumulated DTW cost matrix
    # ------------------------------------------------------------
    cost = torch.full(
        (W1 + 1, W2 + 1),
        float("inf"),
        device=D.device,
        dtype=D.dtype,
    )

    cost[0, 0] = 0.0

    # ------------------------------------------------------------
    # 5. Backpointer matrix
    #
    # 0 = diagonal
    # 1 = vertical
    # 2 = horizontal
    # ------------------------------------------------------------
    backptr = torch.zeros(
        (W1 + 1, W2 + 1),
        device=D.device,
        dtype=torch.int8,
    )

    # ------------------------------------------------------------
    # 6. Dynamic-programming DTW recursion
    # ------------------------------------------------------------
    for i in range(1, W1 + 1):
        for j in range(1, W2 + 1):
            candidates = torch.stack([
                cost[i - 1, j - 1],  # diagonal
                cost[i - 1, j],      # vertical
                cost[i, j - 1],      # horizontal
            ])

            min_cost, min_index = torch.min(candidates, dim=0)

            cost[i, j] = D[i - 1, j - 1] + min_cost
            backptr[i, j] = min_index.to(torch.int8)
    # ------------------------------------------------------------
    # 7. Recover the optimal warping path
    # ------------------------------------------------------------
    i = W1
    j = W2
    path_length = 0

    while i > 0 or j > 0:

        path_length += 1

        direction = int(backptr[i, j].item())

        if direction == 0:
            # Diagonal: (i-1, j-1)
            i -= 1
            j -= 1

        elif direction == 1:
            # Vertical: (i-1, j)
            i -= 1

        elif direction == 2:
            # Horizontal: (i, j-1)
            j -= 1

        else:
            raise RuntimeError(
                f"Invalid DTW backpointer: {direction}"
            )

    # ------------------------------------------------------------
    # 8. Final accumulated DTW cost
    # ------------------------------------------------------------
    dtw_cost = cost[W1, W2]

    # ------------------------------------------------------------
    # 9. Convert path length to tensor
    # ------------------------------------------------------------
    path_length_tensor = torch.tensor(
        path_length,
        device=D.device,
        dtype=D.dtype,
    )

    # ------------------------------------------------------------
    # 10. Length-normalized DTW cost
    # ------------------------------------------------------------
    normalized_cost = dtw_cost / path_length_tensor

    # ------------------------------------------------------------
    # 11. Bounded exponential similarity
    #
    # A = exp(-normalized DTW cost)
    # ------------------------------------------------------------
    similarity = torch.exp(-normalized_cost)

    return dtw_cost, path_length_tensor, similarity


def build_dtw_adjacency(
    X: torch.Tensor,
    self_loops: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Build a dense DTW-based similarity adjacency matrix.

    Parameters
    ----------
    X : torch.Tensor
        Node time-series windows with shape [N, W, F] or [N, W].

    self_loops : bool, default=False
        Whether to retain the diagonal entries.
        If True, A[i, i] = 1 because the DTW cost of a sequence
        with itself is zero.

    Returns
    -------
    dict[str, torch.Tensor]
        ``A_dtw``:
            Dense DTW similarity matrix with shape [N, N].

            A_dtw[i, j] =
                exp(-DTW_cost(i,j) / path_length(i,j))

        Values are bounded in (0, 1] before optional removal
        of self-loops.
    """
    # ------------------------------------------------------------
    # 1. Convert univariate input [N, W] to [N, W, 1]
    # ------------------------------------------------------------
    if X.ndim == 2:
        X = X.unsqueeze(-1)

    if X.ndim != 3:
        raise ValueError(
            "X must have shape [N, W, F] or [N, W]."
        )

    N, W, F = X.shape

    # ------------------------------------------------------------
    # 2. Allocate similarity adjacency
    # ------------------------------------------------------------
    A_dtw = torch.zeros(
        (N, N),
        device=X.device,
        dtype=torch.float32,
    )

    # ------------------------------------------------------------
    # 3. Compute only the upper triangular part.
    #
    # DTW similarity is symmetric:
    #
    # A[i,j] = A[j,i]
    # ------------------------------------------------------------
    start_j = 0 if self_loops else 1

    for i in range(N):

        for j in range(i + start_j, N):

            _, _, similarity = dtw_distance(
                X[i],
                X[j],
            )

            similarity = similarity.to(A_dtw.dtype)

            A_dtw[i, j] = similarity

            if i != j:
                A_dtw[j, i] = similarity

    # ------------------------------------------------------------
    # 4. Remove self-loops if requested
    #
    # Otherwise the diagonal naturally equals 1.
    # ------------------------------------------------------------
    if not self_loops:
        A_dtw.fill_diagonal_(0.0)

    return {
        "A_dtw": A_dtw
    }


def build_dtw_graphs_from_timeseries(
    X: torch.Tensor,
    L: int = 10,
    k: int = 5,
    self_loops: bool = False,
    topk_sym: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Build a temporal sequence of DTW similarity graphs.

    Parameters
    ----------
    X : torch.Tensor
        Multivariate time series with shape [T, N, F].

    L : int, default=10
        Maximum historical window length.

    k : int, default=5
        Number of neighbors retained by the optional Top-K operation.

    self_loops : bool, default=False
        Whether to retain self-loops.

    topk_sym : bool, default=False
        Whether the Top-K graph should be symmetrized.


    Returns
    -------
    dict[str, torch.Tensor]
        ``A_dtw``:
            Dense DTW similarity graphs with shape [T, N, N].

        ``A_topk``:
            Top-K version of ``A_dtw`` with shape [T, N, N].

    Notes
    -----
    For early timesteps, when fewer than L observations are available,
    the function uses all available history instead of artificial padding.

    The effective historical window length is therefore:

        L_t = min(L, t + 1)

    At t >= L - 1, the window has exactly L observations.
    """
    # ------------------------------------------------------------
    # 1. Validate input
    # ------------------------------------------------------------
    if X.ndim != 3:
        raise ValueError(
            "X must have shape [T, N, F]."
        )

    if L < 1:
        raise ValueError(
            "L must be a positive integer."
        )

    T, N, F = X.shape

    if k < 1:
        raise ValueError(
            "k must be a positive integer."
        )

    if k >= N:
        raise ValueError(
            f"k must be smaller than the number of nodes N={N}."
        )

    # ------------------------------------------------------------
    # 2. Allocate temporal graph sequence
    # ------------------------------------------------------------
    A_dtw = torch.zeros(
        (T, N, N),
        device=X.device,
        dtype=torch.float32,
    )

    # ------------------------------------------------------------
    # 3. Optional progress iterator
    # ------------------------------------------------------------
    if tqdm is not None:
        iterator = tqdm(
            range(T),
            desc="Building DTW graphs",
            leave=True,
        )
    else:
        iterator = range(T)

    # ------------------------------------------------------------
    # 4. Construct one DTW graph at each timestep
    # ------------------------------------------------------------
    for t in iterator:

        start = max(0, t - L + 1)

        window = X[start:t + 1]

        # Current shape:
        # [W_t, N, F]
        #
        # Required by build_dtw_adjacency:
        # [N, W_t, F]
        # --------------------------------------------------------
        window = window.permute(1, 0, 2)

        result = build_dtw_adjacency(
            window,
            self_loops=self_loops,
        )

        A_dtw[t] = result["A_dtw"]

    # ------------------------------------------------------------
    # 5. Optional Top-K sparsification with normalization (row-wise or symmetric)
    # ------------------------------------------------------------
    A_topk = topk_row(
        A_dtw,
        k=k,
        sym=topk_sym,
        eps=1e-8,
        preserve_diagonal=self_loops,
    )

    return {
        "A_dtw": A_dtw,
        "A_topk": A_topk,
    }
#-------------------------------------------------------------------
#-----------------------Wind Graph building-------------------------
#-------------------------------------------------------------------

class WindAdjacency(nn.Module):
    """
    Build an incoming wind adjacency from static geometry and wind features.

    Input:
        wind_feats: [N, F] or [B, N, F]

    Output:
        A_in[dst, src] = influence from source src to destination dst.
    """

    def __init__(
        self,
        D_ij: torch.Tensor,
        Theta_ij: torch.Tensor,
        distance_scale: float = 150.0,
        direction_scale: float = 1.0,
        cone_half_angle: float | None = None,
        wind_speed_pos: int = 0,
        wind_dir_pos: int = 1,
        wind_speed_scale: float = 5.0,
        cloud_cover_pos: int | None = None,
        cloud_cover_scale: float = 0.5,
    ) -> None:
        super().__init__()

        if distance_scale <= 0:
            raise ValueError("distance_scale must be positive.")

        if direction_scale <= 0:
            raise ValueError("direction_scale must be positive.")

        if wind_speed_scale <= 0:
            raise ValueError("wind_speed_scale must be positive.")

        if cloud_cover_scale <= 0:
            raise ValueError("cloud_cover_scale must be positive.")

        self.register_buffer("D_ij", D_ij)
        self.register_buffer("Theta_ij", Theta_ij)

        self.distance_scale = float(distance_scale)
        self.direction_scale = float(direction_scale)
        self.cone_half_angle = cone_half_angle

        self.wind_speed_pos = wind_speed_pos
        self.wind_dir_pos = wind_dir_pos
        self.wind_speed_scale = float(wind_speed_scale)

        self.cloud_cover_pos = cloud_cover_pos
        self.cloud_cover_scale = float(cloud_cover_scale)

    @staticmethod
    def angdiff(
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Return signed angular difference in [-pi, pi)."""
        return (a - b + torch.pi) % (2.0 * torch.pi) - torch.pi

    def forward(
        self,
        wind_feats: torch.Tensor,
        self_loops: bool = False,
    ) -> torch.Tensor:
        """
        Construct incoming wind adjacency.

        Returns:
            [N, N] for input [N, F]
            [B, N, N] for input [B, N, F]
        """

        # ---------------------------------------------------------
        # Handle input shape
        # ---------------------------------------------------------
        if wind_feats.dim() == 2:
            wind_feats = wind_feats.unsqueeze(0)
            squeeze = True
        elif wind_feats.dim() == 3:
            squeeze = False
        else:
            raise ValueError(
                "wind_feats must have shape [N, F] or [B, N, F]."
            )

        B, N, _ = wind_feats.shape

        # ---------------------------------------------------------
        # Extract wind features
        # ---------------------------------------------------------
        wind_speed = wind_feats[..., self.wind_speed_pos]
        wind_dir = wind_feats[..., self.wind_dir_pos]

        # Meteorological direction: from -> movement direction: to
        wind_to = (
            wind_dir + torch.pi
        ) % (2.0 * torch.pi)

        # ---------------------------------------------------------
        # Expand static geometry
        # ---------------------------------------------------------
        D_ij = self.D_ij.unsqueeze(0).expand(B, N, N)
        Theta_ij = self.Theta_ij.unsqueeze(0).expand(B, N, N)

        # ---------------------------------------------------------
        # Directional alignment
        # ---------------------------------------------------------
        ang = self.angdiff(
            Theta_ij,
            wind_to.unsqueeze(-1),
        )

        align = torch.cos(ang).clamp(min=0.0)

        # ---------------------------------------------------------
        # Kernel 1: distance
        # ---------------------------------------------------------
        distance_kernel = torch.exp(
            -D_ij / self.distance_scale
        )

        # ---------------------------------------------------------
        # Kernel 2: directional alignment
        # ---------------------------------------------------------
        direction_kernel = torch.exp(
            -(1.0 - align) / self.direction_scale
        )

        # ---------------------------------------------------------
        # Kernel 3: wind speed
        # ---------------------------------------------------------
        wind_speed = torch.clamp(
            wind_speed,
            min=0.0,
        )

        speed_kernel = 1.0 - torch.exp(
            -wind_speed / self.wind_speed_scale
        )

        kernels = [
            distance_kernel,
            direction_kernel,
            speed_kernel.unsqueeze(-1),
        ]

        # ---------------------------------------------------------
        # Optional Kernel 4: cloud cover
        # ---------------------------------------------------------
        if self.cloud_cover_pos is not None:
            cloud_cover = wind_feats[
                ..., self.cloud_cover_pos
            ]

            cloud_cover = torch.clamp(
                cloud_cover,
                min=0.0,
                max=1.0,
            )

            cloud_kernel = 1.0 - torch.exp(
                -cloud_cover / self.cloud_cover_scale
            )

            kernels.append(
                cloud_kernel.unsqueeze(-1)
            )

        # ---------------------------------------------------------
        # Equal geometric mean
        #
        # A_ij = (K1 * K2 * ... * KM)^(1/M)
        # ---------------------------------------------------------
        base = torch.ones_like(distance_kernel)

        for kernel in kernels:
            base = base * kernel

        num_kernels = len(kernels)

        base = base.pow(
            1.0 / num_kernels
        )

        # ---------------------------------------------------------
        # Optional wind cone
        # ---------------------------------------------------------
        if self.cone_half_angle is not None:
            cone_mask = (
                ang.abs() <= self.cone_half_angle
            )

            base = base * cone_mask.to(
                base.dtype
            )

        # ---------------------------------------------------------
        # Convert source -> destination
        # to incoming adjacency A_in[dst, src]
        # ---------------------------------------------------------
        base = base.transpose(-1, -2)

        # ---------------------------------------------------------
        # Self loops
        # ---------------------------------------------------------
        if self_loops:
            base.diagonal(
                dim1=-2,
                dim2=-1,
            ).fill_(1.0)
        else:
            base.diagonal(
                dim1=-2,
                dim2=-1,
            ).zero_()

        # ---------------------------------------------------------
        # Return raw adjacency
        # ---------------------------------------------------------
        A = base

        if squeeze:
            A = A[0]

        return A


def _safe_quantile(
    x: torch.Tensor,
    q: float,
    name: str,
    max_samples: int = 1_000_000,
) -> torch.Tensor:
    """
    Estimate a quantile from at most ``max_samples`` randomly selected values.

    The input is flattened and sampled before ``torch.quantile`` is called,
    preventing quantile computation from operating on very large tensors.

    Args:
        x: Input tensor containing scalar observations.
        q: Quantile in [0, 1].
        name: Name used in the diagnostic message.
        max_samples: Maximum number of values passed to torch.quantile.

    Returns:
        Scalar tensor containing the estimated quantile.
    """
    x = x.reshape(-1)
    total_samples = x.numel()

    if total_samples == 0:
        raise ValueError(f"{name} contains no valid samples.")

    if total_samples > max_samples:
        indices = torch.randint(
            low=0,
            high=total_samples,
            size=(max_samples,),
            device=x.device,
        )
        x = x.index_select(0, indices)
    else:
        indices = None

    return torch.quantile(x, q)


def estimate_wind_kernel_scales(
    D_ij: torch.Tensor,
    Theta_ij: torch.Tensor,
    wind_sp: torch.Tensor,
    wind_dir: torch.Tensor,
    tcc: torch.Tensor | None = None,
    distance_quantile: float = 0.50,
    direction_quantile: float = 0.50,
    wind_speed_quantile: float = 0.50,
    cloud_cover_quantile: float = 0.50,
    max_samples: int = 1_000_000,
) -> dict[str, float]:
    """
    Estimate fixed kernel scales for wind/cloud graph construction.

    Expected input shapes:
        D_ij:    [N, N]
        Theta_ij:[N, N]
        wind_sp: [T, N]
        wind_dir:[T, N]
        tcc:     [T, N] or None

    The scale estimates are obtained from bounded random samples so that
    the procedure remains memory-safe for long time series.

    Returns:
        Dictionary containing:
            distance_scale
            direction_scale
            wind_speed_scale
            cloud_cover_scale (if tcc is provided)
    """

    # ---------------------------------------------------------
    # Convert to floating-point tensors
    # ---------------------------------------------------------
    D_ij = D_ij.float()
    Theta_ij = Theta_ij.float()
    wind_sp = wind_sp.float()
    wind_dir = wind_dir.float()

    if tcc is not None:
        tcc = tcc.float()

    # ---------------------------------------------------------
    # Validate geometry
    # ---------------------------------------------------------
    if D_ij.ndim != 2 or D_ij.shape[0] != D_ij.shape[1]:
        raise ValueError(
            f"D_ij must have shape [N, N], got {D_ij.shape}."
        )

    if Theta_ij.shape != D_ij.shape:
        raise ValueError(
            "Theta_ij must have the same shape as D_ij."
        )

    node_count = D_ij.shape[0]

    if node_count < 2:
        raise ValueError(
            "At least two stations are required."
        )

    # ---------------------------------------------------------
    # Validate temporal wind features
    # ---------------------------------------------------------
    if wind_sp.ndim != 2 or wind_dir.ndim != 2:
        raise ValueError(
            "wind_sp and wind_dir must have shape [T, N]."
        )

    if wind_sp.shape != wind_dir.shape:
        raise ValueError(
            f"wind_sp and wind_dir must have the same shape; "
            f"got {wind_sp.shape} and {wind_dir.shape}."
        )

    if wind_sp.shape[1] != node_count:
        raise ValueError(
            f"Wind features contain {wind_sp.shape[1]} stations, "
            f"but geometry contains {node_count} stations."
        )

    time_count = wind_sp.shape[0]

    if time_count == 0:
        raise ValueError(
            "Wind features contain no time instances."
        )

    # ---------------------------------------------------------
    # Validate cloud cover
    # ---------------------------------------------------------
    if tcc is not None:
        if tcc.ndim != 2:
            raise ValueError(
                f"tcc must have shape [T, N], got {tcc.shape}."
            )

        if tcc.shape != wind_sp.shape:
            raise ValueError(
                f"tcc must have the same shape as wind features; "
                f"got {tcc.shape} and {wind_sp.shape}."
            )

    # ---------------------------------------------------------
    # Validate quantiles
    # ---------------------------------------------------------
    quantiles = (
        distance_quantile,
        direction_quantile,
        wind_speed_quantile,
        cloud_cover_quantile,
    )

    if any(not 0.0 <= q <= 1.0 for q in quantiles):
        raise ValueError(
            "All quantiles must be in the interval [0, 1]."
        )

    if max_samples <= 0:
        raise ValueError(
            "max_samples must be positive."
        )

    # =========================================================
    # 1. Distance scale
    # =========================================================
    #
    # D_ij is symmetric, therefore only unique unordered pairs
    # are used.
    # =========================================================

    upper_mask = torch.triu(
        torch.ones_like(
            D_ij,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    distances = D_ij[upper_mask]
    distances = distances.clamp(min=0.0)

    if distances.numel() == 0:
        raise ValueError(
            "D_ij contains no positive off-diagonal distances."
        )

    distance_scale = _safe_quantile(
        distances,
        distance_quantile,
        "Distance",
        max_samples=max_samples,
    )

    # =========================================================
    # 2. Direction scale
    # =========================================================
    #
    # wind_dir is [T,N].
    #
    # Sample time steps, then include every directed off-diagonal
    # pair from each selected time step.
    # =========================================================

    wind_to = (
        wind_dir + torch.pi
    ) % (2.0 * torch.pi)

    total_direction_pairs = (
        time_count
        * node_count
        * (node_count - 1)
    )
    if total_direction_pairs == 0:
        raise ValueError(
            "No valid directed station pairs are available."
        )
    pairs_per_time = node_count * (node_count - 1)
    sampled_time_count = min(time_count, max_samples // pairs_per_time)
    if sampled_time_count == 0:
        raise ValueError(
            f"max_samples must be at least {pairs_per_time:,} to include "
            "all directed pairs from one time step."
        )

    sampled_time_indices = torch.randperm(
        time_count,
        device=wind_sp.device,
    )[:sampled_time_count]
    selected_wind_to = wind_to.index_select(0, sampled_time_indices)
    angular_difference = WindAdjacency.angdiff(
        Theta_ij.unsqueeze(0),
        selected_wind_to.unsqueeze(-1),
    )
    off_diagonal_mask = ~torch.eye(
        node_count,
        dtype=torch.bool,
        device=Theta_ij.device,
    )
    directional_error = (
        1.0 - torch.cos(angular_difference[:, off_diagonal_mask]).clamp(min=0.0)
    )
    print(
        f"Direction scale estimation: selected {sampled_time_count:,} "
        f"of {time_count:,} time steps"
    )

    direction_scale = _safe_quantile(
        directional_error,
        direction_quantile,
        "Direction",
        max_samples=max_samples,
    )

    # =========================================================
    # 3. Wind-speed scale
    # =========================================================

    wind_speed_scale = _safe_quantile(
        wind_sp.clamp(min=0.0),
        wind_speed_quantile,
        "Wind speed",
        max_samples=max_samples,
    )

    # =========================================================
    # Assemble scales
    # =========================================================

    scales: dict[str, float] = {
        "distance_scale": float(distance_scale),
        "direction_scale": float(direction_scale),
        "wind_speed_scale": float(wind_speed_scale),
    }

    # =========================================================
    # 4. Cloud-cover scale
    # =========================================================

    if tcc is not None:

        cloud_cover = tcc.clamp(
            min=0.0,
            max=1.0,
        )

        cloud_cover_scale = _safe_quantile(
            cloud_cover,
            cloud_cover_quantile,
            "Cloud cover",
            max_samples=max_samples,
        )

        scales["cloud_cover_scale"] = float(
            cloud_cover_scale
        )

    return scales


def build_wind_cloud_adjacency(
    D_ij: torch.Tensor,
    Theta_ij: torch.Tensor,
    wind_sp: torch.Tensor,
    wind_dir: torch.Tensor,
    tcc: torch.Tensor | None = None,
    distance_scale: float | None = None,
    direction_scale: float | None = None,
    cone_half_angle: float | None = None,
    wind_speed_scale: float | None = None,
    cloud_cover_scale: float | None = None,
    k: int = 5,
    self_loops: bool = False,
    topk_sym: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Build an incoming wind/cloud adjacency from node-level wind features.

    Parameters
    ----------
    D_ij : [N, N] distance matrix.
    Theta_ij : [N, N] bearing matrix (radians).
    wind_sp : [N] or [B, N] tensor of wind speeds.
    wind_dir : [N] or [B, N] tensor of wind directions in radians.
    tcc : [N] or [B, N] tensor of cloud cover values. Optional.
    distance_scale : float | None
        Distance decay scale. If None, estimate it from ``D_ij``.
    direction_scale : float | None
        Wind alignment sharpness. If None, estimate it from ``Theta_ij``.
    cone_half_angle : float | None
        If set, restricts edges to nodes within the wind cone.
    wind_speed_scale : float | None
        Wind-speed response scale. If None, estimate it from ``wind_sp``.
    cloud_cover_scale : float | None
        Cloud-cover response scale. If None and ``tcc`` is provided,
        estimate it from ``tcc``.
    k : int
        Number of neighbors to keep in ``A_topk``.
    self_loops : bool
        Whether to keep self-loops in the adjacency.
    topk_sym : bool
        If True, symmetrize the top-k adjacency.

    Returns
    -------
    dict[str, torch.Tensor]
        ``A_wind[dst, src]`` is the weight of the edge from ``src`` to ``dst``.
        Contains dense ``A_wind`` and top-k ``A_topk``, each
        with shape ``[N, N]`` for unbatched input or ``[B, N, N]`` for batched
        input. ``A_topk`` is row-normalized unless ``topk_sym`` is true, in
        which case it is symmetrized and degree-normalized.
    """
    wind_sp = wind_sp.float()
    wind_dir = wind_dir.float()

    if wind_sp.dim() == 1:
        wind_sp = wind_sp.unsqueeze(0)
    if wind_dir.dim() == 1:
        wind_dir = wind_dir.unsqueeze(0)

    needs_estimation = (
        distance_scale is None
        or direction_scale is None
        or wind_speed_scale is None
        or (tcc is not None and cloud_cover_scale is None)
    )
    estimated_scales = (
        estimate_wind_kernel_scales(
            D_ij=D_ij,
            Theta_ij=Theta_ij,
            wind_sp=wind_sp,
            wind_dir=wind_dir,
            tcc=tcc,
        )
        if needs_estimation
        else {}
    )

    distance_scale = (
        estimated_scales["distance_scale"]
        if distance_scale is None
        else distance_scale
    )
    direction_scale = (
        estimated_scales["direction_scale"]
        if direction_scale is None
        else direction_scale
    )
    wind_speed_scale = (
        estimated_scales["wind_speed_scale"]
        if wind_speed_scale is None
        else wind_speed_scale
    )
    if tcc is not None:
        cloud_cover_scale = (
            estimated_scales["cloud_cover_scale"]
            if cloud_cover_scale is None
            else cloud_cover_scale
        )
    else:
        cloud_cover_scale = 0.5 if cloud_cover_scale is None else cloud_cover_scale

    active_scales = {
        "distance_scale": float(distance_scale),
        "direction_scale": float(direction_scale),
        "wind_speed_scale": float(wind_speed_scale),
    }
    if tcc is not None:
        active_scales["cloud_cover_scale"] = float(cloud_cover_scale)
    print(f"Wind kernel scales: {active_scales}")

    if tcc is not None:
        tcc = tcc.float()
        if tcc.dim() == 1:
            tcc = tcc.unsqueeze(0)
        wind_feats = torch.stack([wind_sp, wind_dir, tcc], dim=-1)
        cloud_cover_pos = 2
    else:
        wind_feats = torch.stack([wind_sp, wind_dir], dim=-1)
        cloud_cover_pos = None

    wind_module = WindAdjacency(
        D_ij,
        Theta_ij,
        distance_scale=distance_scale,
        direction_scale=direction_scale,
        cone_half_angle=cone_half_angle,
        wind_speed_pos=0,
        wind_dir_pos=1,
        cloud_cover_pos=cloud_cover_pos,
        wind_speed_scale=wind_speed_scale,
        cloud_cover_scale=cloud_cover_scale,
    )

    A = wind_module(wind_feats, self_loops=self_loops)
    A_topk = topk_row(A, k=k, sym=topk_sym, eps=1e-8, preserve_diagonal=self_loops)
    # A_topk is row-normalized unless topk_sym=True.

    out = {
        "A_wind": A,
        "A_topk": A_topk,
    }

    return out