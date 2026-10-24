"""Plain-script checks for folding the time axis into the GAT batch."""

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Src.MSRL import SpatialGATv2EncoderBatched
from test_current_state_anchor import _pipeline_class


GRAPH_NAMES = ("g_static", "g_wind", "g_semantic")


def _graph_inputs(static_graphs: bool, batch: int, time: int, nodes: int):
    base = torch.rand(nodes, nodes)
    base.fill_diagonal_(0)
    dynamic = torch.rand(batch, time, nodes, nodes)
    dynamic.diagonal(dim1=-2, dim2=-1).zero_()
    spatial_graphs = {
        name: base if static_graphs else dynamic
        for name in GRAPH_NAMES
    }
    spatial_graphs.update({
        "static_graph:A_stat": base,
        "wind_cloud_graphs:A_wind": dynamic,
        "dtw_graphs:A_dtw": dynamic,
    })
    return spatial_graphs


def _make_pipeline(pipeline_cls, spatial_chunk=None):
    return pipeline_cls(
        input_features=3,
        graph_names=list(GRAPH_NAMES),
        horizon_count=2,
        spatial_hidden_dim=4,
        temporal_hidden_dim=4,
        heads=2,
        dropout=0.0,
        cov_dim=5,
        use_cov_queries=True,
        spatial_chunk=spatial_chunk,
    )


def _capture_spatial_sequence(model, x_seq, graphs, cov_future):
    captured = []
    hook = model.temporal_encoder.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach())
    )
    try:
        result = model(x_seq, graphs, cov_future)
    finally:
        hook.remove()
    return captured[0], result


def _old_looped_spatial(model, x_seq, graphs):
    fused_steps = []
    for time_index in range(x_seq.shape[1]):
        adjacencies = [
            graphs[name] if graphs[name].ndim == 2 else graphs[name][:, time_index]
            for name in model.graph_names
        ]
        embeddings = model.spatial_encoder(x_seq[:, time_index], adjacencies)
        fused_steps.append(model.fusion(embeddings))
    return torch.stack(fused_steps, dim=1)


@pytest.mark.parametrize("static_graphs", [True, False])
def test_folded_spatial_sequence_matches_old_loop(static_graphs):
    torch.manual_seed(5)
    pipeline_cls = _pipeline_class()
    model = _make_pipeline(pipeline_cls).eval()
    batch, time, nodes, features = 2, 6, 5, 3
    x_seq = torch.randn(batch, nodes, time, features).transpose(1, 2)
    assert not x_seq.is_contiguous()
    graphs = _graph_inputs(static_graphs, batch, time, nodes)
    cov_future = torch.randn(batch, 2, nodes, 5)

    expected = _old_looped_spatial(model, x_seq, graphs)
    actual, _ = _capture_spatial_sequence(model, x_seq, graphs, cov_future)

    assert actual.shape == (batch, time, nodes, 4)
    assert torch.allclose(actual, expected, atol=1e-5)


def _old_loop_edges(adjacency, batch_size, num_nodes):
    if adjacency.ndim == 2:
        adjacency = adjacency.unsqueeze(0).expand(batch_size, -1, -1)
    all_edges, all_weights = [], []
    for batch_index in range(batch_size):
        dst, src = (adjacency[batch_index] != 0).nonzero(as_tuple=True)
        if dst.numel():
            offset = batch_index * num_nodes
            all_edges.append(torch.stack([src + offset, dst + offset]))
            all_weights.append(adjacency[batch_index, dst, src])
    if not all_edges:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=adjacency.dtype)
    return torch.cat(all_edges, dim=1), torch.cat(all_weights)


def _sort_edges(edge_index, edge_weights):
    key = edge_index[0] * (int(edge_index.max().item()) + 1) + edge_index[1]
    order = key.argsort()
    return edge_index[:, order], edge_weights[order]


def test_vectorized_edge_conversion_matches_old_loop_and_handles_empty_graph():
    torch.manual_seed(7)
    batch, nodes = 3, 5
    adjacency = torch.randint(0, 2, (batch, nodes, nodes)).float()
    adjacency *= torch.rand_like(adjacency)
    actual = SpatialGATv2EncoderBatched._dense_adjacency_to_edges(adjacency, batch, nodes)
    expected = _old_loop_edges(adjacency, batch, nodes)
    actual_edges, actual_weights = _sort_edges(*actual)
    expected_edges, expected_weights = _sort_edges(*expected)
    assert torch.equal(actual_edges, expected_edges)
    assert torch.allclose(actual_weights, expected_weights)

    empty_edges, empty_weights = SpatialGATv2EncoderBatched._dense_adjacency_to_edges(
        torch.zeros(nodes, nodes), batch, nodes
    )
    assert empty_edges.shape == (2, 0)
    assert empty_edges.dtype == torch.long
    assert empty_weights.shape == (0,)
    assert empty_weights.dtype == torch.float32


def test_spatial_chunk_matches_single_pass_and_preserves_gradients():
    torch.manual_seed(9)
    pipeline_cls = _pipeline_class()
    full = _make_pipeline(pipeline_cls).eval()
    chunked = _make_pipeline(pipeline_cls, spatial_chunk=4).eval()
    chunked.load_state_dict(copy.deepcopy(full.state_dict()))
    batch, time, nodes, features = 2, 6, 5, 3
    x_seq = torch.randn(batch, time, nodes, features)
    graphs = _graph_inputs(False, batch, time, nodes)
    cov_future = torch.randn(batch, 2, nodes, 5)

    full_sequence, _ = _capture_spatial_sequence(full, x_seq, graphs, cov_future)
    chunk_sequence, _ = _capture_spatial_sequence(chunked, x_seq, graphs, cov_future)
    assert torch.allclose(full_sequence, chunk_sequence, atol=1e-5)

    output = chunked(x_seq, graphs, cov_future)["forecast"]
    output.square().mean().backward()
    encoder_grads = [parameter.grad for parameter in chunked.spatial_encoder.parameters()]
    fusion_grads = [parameter.grad for parameter in chunked.fusion.parameters()]
    assert encoder_grads and all(grad is not None and torch.isfinite(grad).all() for grad in encoder_grads)
    assert fusion_grads and all(grad is not None and torch.isfinite(grad).all() for grad in fusion_grads)


if __name__ == "__main__":
    test_folded_spatial_sequence_matches_old_loop(static_graphs=True)
    test_folded_spatial_sequence_matches_old_loop(static_graphs=False)
    test_vectorized_edge_conversion_matches_old_loop_and_handles_empty_graph()
    test_spatial_chunk_matches_single_pass_and_preserves_gradients()
    print("spatial time folding tests passed")
