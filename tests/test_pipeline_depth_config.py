"""Plain-script checks for configurable temporal depth and LGEE history."""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_current_state_anchor import _pipeline_class


GRAPH_NAMES = ["static_graph:A_topk", "wind_cloud_graphs:A_topk", "dtw_graphs:A_dtw"]


def _pipeline_kwargs(pipeline_cls, **overrides):
    kwargs = {
        "input_features": 3,
        "graph_names": GRAPH_NAMES,
        "horizon_count": 2,
        "spatial_hidden_dim": 4,
        "temporal_hidden_dim": 4,
        "heads": 2,
        "dropout": 0.0,
        "cov_dim": 5,
        "use_cov_queries": True,
    }
    kwargs.update(overrides)
    return pipeline_cls(**kwargs)


def _inputs(batch=1, time=32, nodes=3, features=3, horizons=2):
    x_seq = torch.randn(batch, time, nodes, features)
    static_graph = torch.eye(nodes)
    dynamic_graph = torch.rand(batch, time, nodes, nodes)
    graphs = {
        "static_graph:A_topk": static_graph,
        "wind_cloud_graphs:A_topk": dynamic_graph,
        "dtw_graphs:A_topk": dynamic_graph,
        "static_graph:A_stat": static_graph,
        "wind_cloud_graphs:A_wind": dynamic_graph,
        "dtw_graphs:A_dtw": dynamic_graph,
    }
    cov_future = torch.randn(batch, horizons, nodes, 5)
    return x_seq, graphs, cov_future


def test_pipeline_forward_supports_multiple_depths_and_matching_history():
    torch.manual_seed(0)
    pipeline_cls = _pipeline_class()
    for num_blocks in (2, 3, 4):
        expected_history = 32 // (2**num_blocks)
        model = _pipeline_kwargs(
            pipeline_cls,
            input_length=32,
            num_blocks=num_blocks,
            dilation_list=[2**i for i in range(num_blocks)],
        )
        x_seq, graphs, cov_future = _inputs()
        result = model(x_seq, graphs, cov_future)
        assert model.history_length == expected_history
        assert model.num_blocks == num_blocks
        assert model.dilation_list == [2**i for i in range(num_blocks)]
        assert model.latent_evolution_encoder.history_length == expected_history
        assert result["forecast"].shape == (1, 2, 3)
        assert torch.isfinite(result["forecast"]).all()


def test_constructor_rejects_nondivisible_length_and_short_dilation_list():
    pipeline_cls = _pipeline_class()
    with pytest.raises(ValueError, match="divisible"):
        _pipeline_kwargs(pipeline_cls, input_length=31, num_blocks=2)
    with pytest.raises(ValueError, match="dilation_list"):
        _pipeline_kwargs(
            pipeline_cls,
            input_length=32,
            num_blocks=3,
            dilation_list=[1, 2],
        )


def test_default_depth_derives_history_and_default_setup_is_preserved():
    pipeline_cls = _pipeline_class()
    default_model = _pipeline_kwargs(pipeline_cls, input_length=32)
    assert default_model.num_blocks == 3
    assert default_model.history_length == 32 // 8

    old_setup = _pipeline_kwargs(pipeline_cls, input_length=336)
    assert old_setup.history_length == 42
    assert old_setup.dilation_list == [1, 2, 4]


def test_forward_reports_mismatched_runtime_history_length():
    pipeline_cls = _pipeline_class()
    model = _pipeline_kwargs(pipeline_cls, input_length=32, num_blocks=2)
    x_seq, graphs, cov_future = _inputs(time=16)
    with pytest.raises(ValueError, match="MTRL output history length mismatch"):
        model(x_seq, graphs, cov_future)


if __name__ == "__main__":
    test_pipeline_forward_supports_multiple_depths_and_matching_history()
    test_constructor_rejects_nondivisible_length_and_short_dilation_list()
    test_default_depth_derives_history_and_default_setup_is_preserved()
    test_forward_reports_mismatched_runtime_history_length()
    print("pipeline depth configuration tests passed")
