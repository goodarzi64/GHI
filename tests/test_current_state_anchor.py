import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Src.anchor import CurrentStateAnchor


def _pipeline_class():
    import pytest

    pytest.importorskip("torch_geometric")
    from Src.FSDG import FutureSpatialDependencyGenerator
    from Src.LGEE import LatentGraphEvolutionEncoder
    from Src.MSRL import SpatialGATv2EncoderBatched
    from Src.MTRL import TemporalContextEncoder
    from Src.fusion_module import SpatialFusionModule
    from Src.forecast_head import ForecastHead
    from Src.FSDP import MultiGraphAdaptivePropagation

    notebook = json.loads((REPO_ROOT / "NoteBooks" / "pipeline.ipynb").read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if "class SpatiotemporalForecastingPipeline" in "".join(cell.get("source", [])))
    namespace = {
        "torch": torch,
        "SpatialGATv2EncoderBatched": SpatialGATv2EncoderBatched,
        "SpatialFusionModule": SpatialFusionModule,
        "TemporalContextEncoder": TemporalContextEncoder,
        "LatentGraphEvolutionEncoder": LatentGraphEvolutionEncoder,
        "FutureSpatialDependencyGenerator": FutureSpatialDependencyGenerator,
        "MultiGraphAdaptivePropagation": MultiGraphAdaptivePropagation,
        "CurrentStateAnchor": CurrentStateAnchor,
        "ForecastHead": ForecastHead,
    }
    exec("".join(cell["source"]), namespace)
    return namespace["SpatiotemporalForecastingPipeline"]


def test_anchor_shape_finiteness_and_initial_near_identity():
    torch.manual_seed(0)
    anchor = CurrentStateAnchor(spatial_dim=5, latent_dim=8)
    z_last = torch.randn(2, 3, 8)
    s_last = torch.randn(2, 3, 5)

    actual = anchor(z_last, s_last)

    assert actual.shape == z_last.shape
    assert torch.isfinite(actual).all()
    # Zero the projection to isolate the initialized gate's sigmoid(-2) ~= 0.12.
    with torch.no_grad():
        anchor.gate.weight.zero_()
        anchor.proj.weight.zero_()
        anchor.proj.bias.fill_(1.0)
    actual = anchor(z_last, s_last)
    assert torch.allclose(actual, z_last + torch.sigmoid(torch.tensor(-2.0)), atol=0.03)


def test_anchor_gradients_flow_to_both_inputs():
    anchor = CurrentStateAnchor(spatial_dim=5, latent_dim=8)
    z_last = torch.randn(2, 3, 8, requires_grad=True)
    s_last = torch.randn(2, 3, 5, requires_grad=True)

    anchor(z_last, s_last).square().mean().backward()

    assert z_last.grad is not None and torch.isfinite(z_last.grad).all()
    assert s_last.grad is not None and torch.isfinite(s_last.grad).all()
    assert torch.count_nonzero(z_last.grad) > 0
    assert torch.count_nonzero(s_last.grad) > 0


def test_pipeline_forward_with_anchor_enabled_and_disabled():
    torch.manual_seed(0)
    pipeline_cls = _pipeline_class()
    graph_names = ["static_graph:A_topk", "wind_cloud_graphs:A_topk", "dtw_graphs:A_topk"]
    batch, time, nodes, features, horizons = 2, 5, 4, 3, 2
    x_seq = torch.randn(batch, time, nodes, features)
    static = torch.eye(nodes)
    dynamic = torch.rand(batch, time, nodes, nodes)
    cov_fut = torch.randn(batch, horizons, nodes, 5)
    graphs = {
        "static_graph:A_topk": static,
        "wind_cloud_graphs:A_topk": dynamic,
        "dtw_graphs:A_topk": dynamic,
        "static_graph:A_stat": static,
        "wind_cloud_graphs:A_wind": dynamic,
        "dtw_graphs:A_dtw": dynamic,
    }

    for use_anchor in (True, False):
        model = pipeline_cls(
            input_features=features,
            graph_names=graph_names,
            horizon_count=horizons,
            spatial_hidden_dim=4,
            temporal_hidden_dim=4,
            heads=2,
            dropout=0.0,
            use_anchor=use_anchor,
            cov_dim=5,
            use_cov_queries=True,
            num_blocks=2,
            input_length=4,
        )
        result = model(x_seq, graphs, cov_fut)
        assert result["forecast"].shape == (batch, horizons, nodes)
        assert torch.isfinite(result["forecast"]).all()
        assert (model.anchor is not None) is use_anchor


if __name__ == "__main__":
    test_anchor_shape_finiteness_and_initial_near_identity()
    test_anchor_gradients_flow_to_both_inputs()
    test_pipeline_forward_with_anchor_enabled_and_disabled()
    print("current state anchor tests passed")
