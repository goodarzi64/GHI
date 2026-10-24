"""Evaluation helpers for reporting forecasts in physical GHI units."""

from __future__ import annotations

from typing import Any

import numpy as np


def invert_ghi_normalization(prediction: Any, mu: float, sigma: float) -> Any:
    """Convert normalized GHI predictions to nonnegative W/m² values."""
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("mu must be finite and sigma must be finite and positive.")
    return (prediction * sigma + mu).clamp(min=0) if hasattr(prediction, "clamp") else np.maximum(
        prediction * sigma + mu, 0
    )


def _regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if prediction.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    error = prediction.astype(np.float64) - target.astype(np.float64)
    ss_total = np.square(target.astype(np.float64) - target.mean(dtype=np.float64)).sum()
    r2 = 1.0 - float(np.square(error).sum()) / float(ss_total) if ss_total > 0 else float("nan")
    return {
        "rmse": float(np.sqrt(np.square(error).mean())),
        "mae": float(np.abs(error).mean()),
        "r2": r2,
    }


def ghi_metrics_wm2(
    prediction_norm: np.ndarray,
    target_raw: np.ndarray,
    mu: float,
    sigma: float,
    daytime_mask: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute all hour and daytime RMSE, MAE, and R² in W/m².

    Inputs have shape ``[B,H,N]``; ``daytime_mask`` is broadcastable to it.
    """
    prediction = np.asarray(invert_ghi_normalization(prediction_norm, mu, sigma))
    target = np.asarray(target_raw)
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction and raw target shapes differ: {prediction.shape} vs {target.shape}.")
    day = np.broadcast_to(np.asarray(daytime_mask, dtype=bool), target.shape)
    return {
        "all_hours": _regression_metrics(prediction, target),
        "daytime": _regression_metrics(prediction[day], target[day]),
    }


def ghi_metrics_by_horizon_wm2(
    prediction_norm: np.ndarray,
    target_raw: np.ndarray,
    mu: float,
    sigma: float,
    daytime_mask: np.ndarray,
) -> list[dict[str, dict[str, float]]]:
    """Compute overall and daytime metrics separately for each horizon."""
    prediction = np.asarray(prediction_norm)
    target = np.asarray(target_raw)
    day = np.broadcast_to(np.asarray(daytime_mask, dtype=bool), target.shape)
    return [
        ghi_metrics_wm2(prediction[:, horizon:horizon + 1], target[:, horizon:horizon + 1],
                        mu, sigma, day[:, horizon:horizon + 1])
        for horizon in range(prediction.shape[1])
    ]
