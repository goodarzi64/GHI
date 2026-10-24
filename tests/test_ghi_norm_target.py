"""Plain-script checks for normalized GHI targets and W/m² reporting."""

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Src.forecast_head import ForecastAndGraphLoss, ForecastHead
from Src.forecast_metrics import ghi_metrics_wm2, invert_ghi_normalization
from Src.data_structure import TemporalSplit, build_windowed_dataset


def test_inversion_round_trip_and_nonnegative_predictions():
    mu, sigma = 250.0, 100.0
    raw = np.array([0.0, 120.0, 450.0], dtype=np.float32)
    normalized = (raw - mu) / sigma
    restored = invert_ghi_normalization(normalized, mu, sigma)
    assert np.allclose(restored, raw)
    assert np.all(invert_ghi_normalization(np.array([-100.0]), mu, sigma) >= 0)


def test_metrics_are_reported_in_watts_per_square_meter():
    mu, sigma = 250.0, 80.0
    target_raw = np.full((2, 3, 4), 300.0, dtype=np.float32)
    target_norm = (target_raw - mu) / sigma
    prediction_norm = target_norm + 1.0
    metrics = ghi_metrics_wm2(
        prediction_norm, target_raw, mu, sigma, np.ones_like(target_raw, dtype=bool)
    )
    assert np.isclose(metrics["all_hours"]["rmse"], sigma)
    assert np.isclose(metrics["all_hours"]["mae"], sigma)


def test_normalized_target_forecast_and_loss_are_finite():
    torch.manual_seed(0)
    head = ForecastHead(channels=4, num_horizons=3)
    loss_fn = ForecastAndGraphLoss(graph_weight=0.0)
    state = torch.randn(2, 3, 5, 4)
    prediction = head(state)
    target_norm = torch.randn(2, 3, 5)
    graph = torch.eye(5).expand(2, 3, 5, 5)
    loss = loss_fn(prediction, target_norm, graph, graph, graph, graph)
    assert prediction.shape == target_norm.shape
    assert torch.isfinite(loss)
    loss.backward()


def test_raw_target_windows_align_with_normalized_targets():
    time_steps, nodes = 12, 2
    tensor = np.zeros((time_steps, nodes, 2), dtype=np.float32)
    tensor[..., 0] = np.arange(time_steps)[:, None] / 10.0
    tensor[..., 1] = np.arange(time_steps)[:, None] * 100.0
    split = TemporalSplit("toy", (0, 6), (6, 9), (9, 12))
    dataset = build_windowed_dataset(
        tensor,
        {},
        split,
        T=2,
        H=1,
        output_mask=np.array([True, False]),
        raw_target_tensor=tensor[..., 1],
    )
    assert dataset.test.y.shape == (1, 1, nodes, 1)
    assert dataset.test.y_raw.shape == (1, 1, nodes)
    assert np.all(dataset.test.y_raw == 1100.0)


if __name__ == "__main__":
    test_inversion_round_trip_and_nonnegative_predictions()
    test_metrics_are_reported_in_watts_per_square_meter()
    test_normalized_target_forecast_and_loss_are_finite()
