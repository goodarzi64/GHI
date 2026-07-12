from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from .GST_Utils import row_normalize, symmetry_normalize, topk_row
import torch.nn as nn

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    tqdm = None


class GeoGeometry:
    def __init__(self, df_geo, device: str = "cpu") -> None:
        self.df_geo = df_geo
        self.device = device
        self._build()

    def _build(self) -> None:
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
    def __init__(self, dist_matrix: torch.Tensor, sigma: torch.Tensor | float | None = None) -> None:
        self.dist_matrix = dist_matrix
        self.sigma = sigma or self._estimate_sigma()

    def _estimate_sigma(self) -> torch.Tensor:
        mask = torch.triu(torch.ones_like(self.dist_matrix), diagonal=1).bool()
        return torch.std(self.dist_matrix[mask])

    def compute(self, self_loops: bool = False) -> torch.Tensor:
        A = torch.exp(- (self.dist_matrix ** 2) / (2 * self.sigma ** 2))
        if not self_loops:
            A.fill_diagonal_(0)
        return A


def build_geo_matrices(df_geo, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Compute distance/theta matrices once and reuse across modules.
    """
    geo = GeoGeometry(df_geo, device=device)
    return {
        "dist_matrix": geo.dist_matrix,
        "theta_matrix": geo.theta_matrix,
    }


def dtw_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute DTW distance between two time series of shape [W, F]."""
    x = x.float()
    y = y.float()

    D = torch.cdist(x, y, p=2)  # [W, W]
    W1, W2 = D.shape

    cost = torch.full((W1 + 1, W2 + 1), float("inf"), device=D.device, dtype=D.dtype)
    cost[0, 0] = 0.0

    for i in range(1, W1 + 1):
        for j in range(1, W2 + 1):
            prev = torch.min(torch.stack([cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]]))
            cost[i, j] = D[i - 1, j - 1] + prev

    return cost[W1, W2]


def build_semantic_adjacency(
    ghi: torch.Tensor,
    tcc: torch.Tensor,
    sigma: float | None = None,
    k: int = 5,
    self_loops: bool = False,
    topk_sym: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Build a semantic graph from history windows using DTW on GHI and TCC.

    Parameters
    ----------
    ghi : torch.Tensor
        Shape [N, W] or [N, W, 1].
    tcc : torch.Tensor
        Shape [N, W] or [N, W, 1].

    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary containing:
        - A_dtw: pairwise DTW distance matrix.
        - A_sim: similarity matrix from Gaussian kernel.
        - A_topk: top-k sparsified similarity adjacency.
        - A_row_norm: row-normalized similarity adjacency.
        - A_sym_norm: symmetric normalized similarity adjacency.
    """
    ghi = ghi.float()
    tcc = tcc.float()

    if ghi.shape != tcc.shape:
        raise ValueError("ghi and tcc must have the same shape [N, W].")

    if ghi.ndim == 3 and ghi.shape[-1] == 1:
        ghi = ghi.squeeze(-1)
    if tcc.ndim == 3 and tcc.shape[-1] == 1:
        tcc = tcc.squeeze(-1)

    N, W = ghi.shape
    X = torch.stack([ghi, tcc], dim=-1)  # [N, W, 2]

    dtw_dist = torch.zeros((N, N), device=X.device, dtype=torch.float32)
    for i in range(N):
        for j in range(i + 1, N):
            dist = dtw_distance(X[i], X[j])
            dtw_dist[i, j] = dist
            dtw_dist[j, i] = dist

    if sigma is None:
        mask = torch.triu(torch.ones_like(dtw_dist), diagonal=1).bool()
        sigma = torch.std(dtw_dist[mask]) if dtw_dist[mask].numel() else torch.tensor(1.0, device=X.device)
    else:
        sigma = torch.tensor(sigma, device=X.device, dtype=torch.float32)

    A_sim = torch.exp(- (dtw_dist ** 2) / (2 * sigma ** 2))
    if not self_loops:
        A_sim.fill_diagonal_(0)

    return {
        "A_dtw": dtw_dist,
        "A_sim": A_sim,
        "A_topk": topk_row(A_sim, k=k, sym=topk_sym, eps=1e-8),
        "A_row_norm": row_normalize(A_sim, eps=1e-8),
        "A_sym_norm": symmetry_normalize(A_sim, eps=1e-8),
    }


def build_static_adjacency(
    dist_matrix: torch.Tensor | None = None,
    df_geo=None,
    device: str = "cpu",
    k: int = 5,
    self_loops: bool = False,
    topk_sym: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Build common static adjacency variants from a distance matrix.

    Returns a dict with:
      - A_raw
      - A_topk
      - A_row_norm
      - A_sym_norm

    Backward compatibility:
      - If `dist_matrix` is None and `df_geo` is provided, matrices are computed
        internally and included in the output.
    """
    geo_mats = None
    if dist_matrix is None:
        if df_geo is None:
            raise ValueError("Provide `dist_matrix` or `df_geo`.")
        geo_mats = build_geo_matrices(df_geo=df_geo, device=device)
        dist_matrix = geo_mats["dist_matrix"]

    kernel = DistanceKernel(dist_matrix, sigma=None)
    A_raw = kernel.compute(self_loops=self_loops)

    out = {
        "A_raw": A_raw,
        "A_topk": topk_row(A_raw, k=k, sym=topk_sym, eps=1e-8),
        "A_row_norm": row_normalize(A_raw, eps=1e-8),
        "A_sym_norm": symmetry_normalize(A_raw, eps=1e-8),
    }
    if geo_mats is not None:
        out["dist_matrix"] = geo_mats["dist_matrix"]
        out["theta_matrix"] = geo_mats["theta_matrix"]
    return out


def build_wind_cloud_adjacency(
    D_ij: torch.Tensor,
    Theta_ij: torch.Tensor,
    wind_sp: torch.Tensor,
    wind_dir: torch.Tensor,
    tcc: torch.Tensor | None = None,
    R: float = 150.0,
    lambda_theta: float = 1.0,
    cone_half_angle: float | None = None,
    cloud_cover_alpha: float = 1.0,
    k: int = 5,
    self_loops: bool = False,
    sparse: bool = False,
    topk_sym: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Build a wind/cloud-informed adjacency from node-level wind features.

    Parameters
    ----------
    D_ij : [N, N] distance matrix.
    Theta_ij : [N, N] bearing matrix (radians).
    wind_sp : [N] or [B, N] tensor of wind speeds.
    wind_dir : [N] or [B, N] tensor of wind directions in radians.
    tcc : [N] or [B, N] tensor of cloud cover values. Optional.
    R : float
        Distance decay scale.
    lambda_theta : float
        Wind alignment sharpness.
    cone_half_angle : float | None
        If set, restricts edges to nodes within the wind cone.
    cloud_cover_alpha : float
        Multiplier for cloud cover adjustment.
    k : int
        Number of neighbors to keep when sparsifying.
    self_loops : bool
        Whether to keep self-loops in the adjacency.
    sparse : bool
        If True, use top-k sparsification inside the wind adjacency module.
    topk_sym : bool
        If True, symmetrize and sparsify the resulting adjacency.

    Returns
    -------
    dict[str, torch.Tensor]
        - A_wind: wind/cloud adjacency [N, N] or [B, N, N].
        - A_row_norm: row-normalized adjacency.
        - A_sym_norm: symmetric normalized adjacency.
        - A_topk: sparsified adjacency if `topk_sym` is True.
    """
    wind_sp = wind_sp.float()
    wind_dir = wind_dir.float()

    if wind_sp.dim() == 1:
        wind_sp = wind_sp.unsqueeze(0)
    if wind_dir.dim() == 1:
        wind_dir = wind_dir.unsqueeze(0)

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
        R=R,
        lambda_theta=lambda_theta,
        cone_half_angle=cone_half_angle,
        wind_speed_pos=0,
        wind_dir_pos=1,
        cloud_cover_pos=cloud_cover_pos,
        cloud_cover_alpha=cloud_cover_alpha,
    )

    A = wind_module(wind_feats, sparse=sparse, k=k, self_loops=self_loops)
    if topk_sym and not sparse:
        A = topk_row(A, k=k, sym=True, eps=1e-8)

    out = {
        "A_wind": A,
        "A_row_norm": row_normalize(A, eps=1e-8),
        "A_sym_norm": symmetry_normalize(A, eps=1e-8),
    }
    if topk_sym:
        out["A_topk"] = topk_row(A, k=k, sym=True, eps=1e-8)

    return out


def build_dtw_adjacency(
    X: torch.Tensor,
    sigma: float | None = None,
    k: int = 5,
    self_loops: bool = False,
    topk_sym: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Build a DTW-based similarity adjacency from multivariate node windows.

    Parameters
    ----------
    X : [N, W, F] or [N, W] tensor.
        Node windows over time.
    sigma : float | None
        Gaussian kernel width. If None, estimated from upper triangle.
    k : int
        Number of neighbors used for optional sparsification.
    self_loops : bool
        Whether to keep self-loops in the similarity matrix.
    topk_sym : bool
        If True, returns an additional symmetric top-k adjacency.

    Returns
    -------
    dict[str, torch.Tensor]
        - A_dtw: DTW distance matrix [N, N].
        - A_sim: Gaussian similarity matrix [N, N].
        - A_row_norm: row-normalized similarity.
        - A_sym_norm: symmetric normalized similarity.
        - A_topk: optional sparsified adjacency if `topk_sym` is True.
    """
    if X.ndim == 2:
        X = X.unsqueeze(-1)
    if X.ndim != 3:
        raise ValueError("X must have shape [N, W, F] or [N, W].")

    N, W, F = X.shape
    dtw_dist = torch.zeros((N, N), device=X.device, dtype=torch.float32)
    for i in range(N):
        for j in range(i + 1, N):
            dist = dtw_distance(X[i], X[j])
            dtw_dist[i, j] = dist
            dtw_dist[j, i] = dist

    if sigma is None:
        mask = torch.triu(torch.ones_like(dtw_dist), diagonal=1).bool()
        sigma = torch.std(dtw_dist[mask]) if dtw_dist[mask].numel() else torch.tensor(1.0, device=X.device)
    else:
        sigma = torch.tensor(sigma, device=X.device, dtype=torch.float32)

    A_sim = torch.exp(- (dtw_dist ** 2) / (2 * sigma ** 2))
    if not self_loops:
        A_sim.fill_diagonal_(0)

    out = {
        "A_dtw": dtw_dist,
        "A_sim": A_sim,
        "A_row_norm": row_normalize(A_sim, eps=1e-8),
        "A_sym_norm": symmetry_normalize(A_sim, eps=1e-8),
    }
    if topk_sym:
        out["A_topk"] = topk_row(A_sim, k=k, sym=True, eps=1e-8)

    return out


def build_dtw_graphs_from_timeseries(
    X: torch.Tensor,
    L: int,
    sigma: float | None = None,
    k: int = 5,
    self_loops: bool = False,
    topk_sym: bool = False,
    progress: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Build a time series of DTW graphs for each timestep.

    Parameters
    ----------
    X : [T, N, F] tensor
        Multivariate time series for N nodes.
    L : int
        Window length used for DTW comparisons.
    sigma : float | None
        Gaussian kernel width.
    k : int
        Number of neighbors for optional top-k sparsification.
    self_loops : bool
        Whether to preserve self-loops.
    topk_sym : bool
        If True, returns symmetric top-k graphs.

    Returns
    -------
    dict[str, torch.Tensor]
        - A_dtw: [T, N, N] DTW distance matrices.
        - A_sim: [T, N, N] similarity matrices.
        - A_row_norm: [T, N, N] row-normalized similarities.
        - A_sym_norm: [T, N, N] symmetric normalized similarities.
        - A_topk: [T, N, N] sparsified graphs if `topk_sym` is True.
    """
    if X.ndim != 3:
        raise ValueError("X must have shape [T, N, F].")
    if L < 1:
        raise ValueError("L must be a positive integer.")

    T, N, F = X.shape
    A_dtw = torch.zeros((T, N, N), device=X.device, dtype=torch.float32)
    A_sim = torch.zeros((T, N, N), device=X.device, dtype=torch.float32)
    A_row_norm = torch.zeros((T, N, N), device=X.device, dtype=torch.float32)
    A_sym_norm = torch.zeros((T, N, N), device=X.device, dtype=torch.float32)
    A_topk = torch.zeros((T, N, N), device=X.device, dtype=torch.float32) if topk_sym else None

    pad_step = X[min(L - 1, T - 1)]
    iterator = (
        tqdm(range(T), desc="Building DTW graphs", leave=True)
        if progress and tqdm is not None
        else range(T)
    )
    for t in iterator:
        if t + 1 >= L:
            window = X[t - L + 1 : t + 1]
        else:
            pad_count = L - (t + 1)
            pad = pad_step.unsqueeze(0).expand(pad_count, N, F)
            window = torch.cat([pad, X[: t + 1]], dim=0)

        result = build_dtw_adjacency(
            window.permute(1, 0, 2),
            sigma=sigma,
            k=k,
            self_loops=self_loops,
            topk_sym=topk_sym,
        )
        A_dtw[t] = result["A_dtw"]
        A_sim[t] = result["A_sim"]
        A_row_norm[t] = result["A_row_norm"]
        A_sym_norm[t] = result["A_sym_norm"]
        if topk_sym:
            A_topk[t] = result["A_topk"]

    output = {
        "A_dtw": A_dtw,
        "A_sim": A_sim,
        "A_row_norm": A_row_norm,
        "A_sym_norm": A_sym_norm,
    }
    if topk_sym:
        output["A_topk"] = A_topk

    return output


class WindAdjacency(nn.Module):
    """
    Build A_wind(t) from static distance/bearing + time-varying wind dir/speed.
    Supports batched input: wind_feats [B, N, F].
    Produces row-stochastic adjacency per batch.
    Important: A_wind[i,j] is how much node i influences node j along wind.
    """

    def __init__(
        self,
        D_ij: torch.Tensor,
        Theta_ij: torch.Tensor,
        R: float = 150.0,
        lambda_theta: float = 1.0,
        cone_half_angle: float | None = None,
        wind_speed_pos: int = 0,
        wind_dir_pos: int = 1,
        cloud_cover_pos: int | None = None,
        cloud_cover_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("D_ij", D_ij)         # [N,N]
        self.register_buffer("Theta_ij", Theta_ij) # [N,N]

        self.R = float(R)
        self.lambda_theta = float(lambda_theta)
        self.cone_half_angle = cone_half_angle

        self.wind_speed_pos = wind_speed_pos
        self.wind_dir_pos = wind_dir_pos
        self.cloud_cover_pos = cloud_cover_pos
        self.cloud_cover_alpha = float(cloud_cover_alpha)

    @staticmethod
    def angdiff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # works with broadcasting (batched or non-batched)
        return (a - b + torch.pi) % (2 * torch.pi) - torch.pi

    def forward(
        self,
        wind_feats: torch.Tensor,
        sparse: bool = False,
        k: int = 5,
        self_loops: bool = False,
    ) -> torch.Tensor:
        # ---------------------------
        # Handle shapes
        # ---------------------------
        if wind_feats.dim() == 2:
            # [N,F] → [1,N,F], later squeeze back
            wind_feats = wind_feats.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B, N, _F = wind_feats.shape

        # ---------------------------
        # Extract wind components
        # ---------------------------
        wind_speed = wind_feats[..., self.wind_speed_pos]  # [B,N]
        wind_dir = wind_feats[..., self.wind_dir_pos]      # [B,N]

        # Convert meteorological (from) → movement (to)
        wind_to = (wind_dir + torch.pi) % (2 * torch.pi)    # [B,N]

        # Expand to [B,N,N]
        dir_mat = wind_to.unsqueeze(-1).expand(B, N, N)     # rows = i

        # Static fields to [B,N,N]
        D_ij = self.D_ij.unsqueeze(0).expand(B, N, N)
        Theta_ij = self.Theta_ij.unsqueeze(0).expand(B, N, N)

        # ---------------------------
        # Alignment term
        # ---------------------------
        # Theta_ij - wind_to[i] → positive if j is downstream from i according to wind
        ang = self.angdiff(Theta_ij, dir_mat)
        align = torch.cos(ang).clamp(min=0.0)               # [B,N,N]

        # ---------------------------
        # Base weight (distance + alignment)
        # ---------------------------
        base = torch.exp(-D_ij / self.R) * torch.exp(align / self.lambda_theta)

        # ---------------------------
        # Cone restriction (optional)
        # ---------------------------
        if self.cone_half_angle is not None:
            cone_mask = (ang.abs() <= self.cone_half_angle)
            base = base * cone_mask.float()

        # ---------------------------
        # Multiply by wind speed_i
        # ---------------------------
        base = base * wind_speed.unsqueeze(-1)              # [B,N,1] → [B,N,N]

        # ---------------------------
        # Multiply by cloud cover if available
        # ---------------------------
        if self.cloud_cover_pos is not None:
            cloud_cover = wind_feats[..., self.cloud_cover_pos]  # [B,N]
            cc_factor = 1.0 + self.cloud_cover_alpha * cloud_cover
            base = base * cc_factor.unsqueeze(-1)

        # ---------------------------
        # self loop connection
        # ---------------------------
        if not self_loops:
            base.diagonal(dim1=-2, dim2=-1).zero_()

        # ---------------------------
        # Sparse or dense?
        # ---------------------------
        if sparse:
            A = []
            for b in range(B):
                A_b = topk_row(base[b], k=k, sym=False, eps=1e-8)     # returns [N,N]
                A.append(A_b)
            A = torch.stack(A, dim=0)                       # [B,N,N]
        else:
            A = base / (base.sum(dim=-1, keepdim=True) + 1e-8)

        # ---------------------------
        # If original input was unbatched → squeeze
        # ---------------------------
        if squeeze:
            A = A[0]

        return A
