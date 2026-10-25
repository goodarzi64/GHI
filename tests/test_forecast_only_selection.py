"""Checks for forecast-only checkpoint selection and loss reporting."""

import ast
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Src.forecast_head import ForecastAndGraphLoss


def test_sample_weighted_forecast_loss_matches_mean_horizon_huber():
    torch.manual_seed(0)
    prediction = torch.randn(5, 4, 3)
    target = torch.randn_like(prediction)
    loss_fn = ForecastAndGraphLoss()

    forecast_loss_total = 0.0
    sample_total = 0
    for indices in (slice(0, 3), slice(3, 5)):
        batch_prediction = prediction[indices]
        batch_target = target[indices]
        batch_size = batch_prediction.shape[0]
        forecast_loss_total += loss_fn._huber_loss(batch_prediction, batch_target).item() * batch_size
        sample_total += batch_size

    accumulated_forecast_loss = forecast_loss_total / sample_total
    difference = (prediction - target).abs()
    huber = torch.where(
        difference <= loss_fn.delta,
        0.5 * difference.square(),
        loss_fn.delta * (difference - 0.5 * loss_fn.delta),
    )
    per_horizon = huber.mean(dim=(0, 2))

    assert torch.allclose(
        torch.tensor(accumulated_forecast_loss), per_horizon.mean(), atol=1e-7
    )


def test_composite_loss_equals_forecast_loss_without_graph_terms():
    torch.manual_seed(1)
    loss_fn = ForecastAndGraphLoss(graph_weight=0.0, sparsity_weight=0.0)
    prediction = torch.randn(2, 3, 4)
    target = torch.randn_like(prediction)
    graph_prediction = torch.rand(2, 3, 4, 4)
    graph_target = torch.rand_like(graph_prediction)

    composite_loss = loss_fn(
        prediction,
        target,
        graph_prediction,
        graph_target,
        graph_prediction,
        graph_target,
    )
    forecast_loss = loss_fn._huber_loss(prediction, target)

    assert torch.allclose(composite_loss, forecast_loss)


def _selection_helper():
    notebook = json.loads((REPO_ROOT / "NoteBooks" / "pipeline.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "def should_update_best" not in source:
            continue
        module = ast.parse(source)
        helper = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "should_update_best"
        )
        namespace = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "pipeline.ipynb", "exec"), namespace)
        return namespace["should_update_best"]
    raise AssertionError("Could not find should_update_best in pipeline.ipynb")


def test_selection_uses_forecast_loss_not_composite_loss():
    should_update_best = _selection_helper()

    # Composite loss improves while forecast loss gets worse: keep the checkpoint.
    prior_composite, current_composite = 0.8, 0.5
    prior_forecast, current_forecast = 0.2, 0.25
    assert current_composite < prior_composite
    assert not should_update_best(current_forecast, prior_forecast)

    # Forecast loss improves while composite loss gets worse: update the checkpoint.
    prior_composite, current_composite = 0.3, 0.6
    prior_forecast, current_forecast = 0.2, 0.15
    assert current_composite > prior_composite
    assert should_update_best(current_forecast, prior_forecast)


if __name__ == "__main__":
    test_sample_weighted_forecast_loss_matches_mean_horizon_huber()
    test_composite_loss_equals_forecast_loss_without_graph_terms()
    test_selection_uses_forecast_loss_not_composite_loss()
    print("forecast-only selection tests passed")
