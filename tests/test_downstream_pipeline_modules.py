import sys

import torch

sys.path.insert(0, 'c:/Users/Mohsen/Documents/GHI')

from Src.forecast_head import ForecastHead
from Src.Graph_build import (
    WindAdjacency,
    build_dtw_adjacency,
    build_dtw_graphs_from_timeseries,
    build_static_adjacency,
    build_wind_cloud_adjacency,
    estimate_wind_kernel_scales,
)
from Src.future_spatial_dependency import FutureSpatialDependencyGenerator
from Src.future_spatial_propagation import CurrentStateRefinement, MultiGraphAdaptivePropagation


def test_static_adjacency_accepts_precomputed_distances():
    distances = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])

    result = build_static_adjacency(dist_matrix=distances, k=1, self_loops=True)

    assert set(result) == {"A_raw", "A_topk"}
    assert all(result[key].shape == (3, 3) for key in result)
    assert torch.allclose(result["A_topk"], result["A_topk"].T)


def test_wind_adjacency_uses_receiver_rows():
    distances = torch.tensor([[0.0, 1.0, 3.0], [2.0, 0.0, 1.0], [3.0, 2.0, 0.0]])
    bearings = torch.zeros_like(distances)
    wind_features = torch.tensor([[1.0, -torch.pi], [2.0, -torch.pi], [3.0, -torch.pi]])
    module = WindAdjacency(distances, bearings, distance_scale=1.0, direction_scale=1.0)

    actual = module(wind_features)
    outgoing_weights = torch.exp(-distances) * torch.exp(torch.ones_like(distances))
    outgoing_weights *= wind_features[:, 0].unsqueeze(1)
    outgoing_weights.fill_diagonal_(0.0)
    expected = outgoing_weights.T
    expected = expected / (expected.sum(dim=-1, keepdim=True) + 1e-8)

    assert torch.allclose(actual, expected)


def test_estimate_wind_kernel_scales_uses_directed_angles():
    distances = torch.tensor([
        [0.0, 2.0, 4.0],
        [2.0, 0.0, 3.0],
        [4.0, 3.0, 0.0],
    ])
    bearings = torch.tensor([
        [0.0, 0.0, 0.0],
        [torch.pi, 0.0, 0.0],
        [torch.pi / 2, torch.pi / 2, 0.0],
    ])
    wind_speed = torch.tensor([1.0, 2.0, 3.0])
    wind_direction = torch.tensor([0.0, torch.pi, torch.pi / 2])
    cloud_cover = torch.tensor([0.2, 0.4, 0.6])

    scales = estimate_wind_kernel_scales(
        distances,
        bearings,
        wind_speed,
        wind_direction,
        cloud_cover,
    )

    assert set(scales) == {
        "distance_scale",
        "direction_scale",
        "wind_speed_scale",
        "cloud_cover_scale",
    }
    assert scales["distance_scale"] == 3.0
    assert scales["wind_speed_scale"] == 2.0
    assert abs(scales["cloud_cover_scale"] - 0.4) < 1e-6
    assert scales["direction_scale"] > 0.0


def test_wind_cloud_builder_returns_dense_and_topk_adjacencies():
    distances = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    bearings = torch.zeros_like(distances)
    wind_speed = torch.ones(2, 3)
    wind_direction = torch.full((2, 3), -torch.pi)

    result = build_wind_cloud_adjacency(
        distances,
        bearings,
        wind_speed,
        wind_direction,
        k=1,
    )

    assert set(result) == {"A_wind", "A_topk"}
    assert result["A_wind"].shape == (2, 3, 3)
    assert result["A_topk"].shape == (2, 3, 3)
    assert (result["A_wind"] != 0).sum(dim=-1).eq(2).all()
    assert (result["A_topk"] != 0).sum(dim=-1).eq(1).all()


def test_wind_cloud_zero_cover_removes_source_edges():
    distances = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    bearings = torch.zeros_like(distances)
    wind_speed = torch.ones(3)
    wind_direction = torch.full((3,), -torch.pi)
    cloud_cover = torch.tensor([0.0, 1.0, 1.0])

    result = build_wind_cloud_adjacency(
        distances,
        bearings,
        wind_speed,
        wind_direction,
        tcc=cloud_cover,
        k=1,
    )

    assert result["A_wind"][..., 0].eq(0).all()


def test_dtw_builders_return_distances_and_optional_topk_only():
    windows = torch.tensor([[[0.0], [1.0]], [[1.0], [2.0]], [[3.0], [4.0]]])
    adjacency = build_dtw_adjacency(windows)

    assert set(adjacency) == {"A_dtw"}
    assert adjacency["A_dtw"].shape == (3, 3)

    timeseries = windows.transpose(0, 1)
    graph_series = build_dtw_graphs_from_timeseries(
        timeseries,
        L=1,
        k=1,
        topk_sym=True,
    )

    assert set(graph_series) == {"A_dtw", "A_topk"}
    assert graph_series["A_dtw"].shape == (2, 3, 3)
    assert graph_series["A_topk"].shape == (2, 3, 3)
    assert torch.allclose(
        graph_series["A_topk"],
        graph_series["A_topk"].transpose(-1, -2),
    )
    assert (graph_series["A_topk"] != 0).sum(dim=-1).ge(1).all()


def test_downstream_pipeline_modules():
    torch.manual_seed(0)

    refinement = CurrentStateRefinement(channels=8, dropout=0.0)
    z_current = torch.randn(2, 5, 8)
    z_refined = refinement(z_current)
    assert z_refined.shape == z_current.shape
    assert torch.isfinite(z_refined).all()

    graph_gen = FutureSpatialDependencyGenerator(latent_dim=8, hidden_dim=16, residual_scale=0.1)
    z_graph = torch.randn(2, 3, 5, 8)
    a_wind_current_dense = torch.rand(2, 5, 5)
    a_sem_current_dense = torch.rand(2, 5, 5)
    a_wind_hat, a_sem_hat = graph_gen(z_graph, a_wind_current_dense, a_sem_current_dense, return_sparse=False)
    assert a_wind_hat.shape == (2, 3, 5, 5)
    assert a_sem_hat.shape == (2, 3, 5, 5)
    assert torch.isfinite(a_wind_hat).all()
    assert torch.isfinite(a_sem_hat).all()

    propagator = MultiGraphAdaptivePropagation(channels=8, num_horizons=3, propagation_steps=2, alpha=0.1, dropout=0.0, use_refinement=False)
    a_phys = torch.rand(5, 5)
    propagated = propagator(z_current, a_phys, a_wind_hat, a_sem_hat)
    assert propagated.shape == (2, 3, 5, 8)
    assert torch.isfinite(propagated).all()

    head = ForecastHead(channels=8, dropout=0.0)
    forecast = head(propagated)
    assert forecast.shape == (2, 3, 5)
    assert torch.isfinite(forecast).all()


def test_propagation_refinement_toggle():
    torch.manual_seed(0)
    z_current = torch.randn(2, 5, 8)
    a_phys = torch.rand(5, 5)
    a_wind_hat = torch.rand(2, 3, 5, 5)
    a_sem_hat = torch.rand(2, 3, 5, 5)

    no_refine = MultiGraphAdaptivePropagation(channels=8, num_horizons=3, propagation_steps=2, alpha=0.1, dropout=0.0, use_refinement=False)
    with_refine = MultiGraphAdaptivePropagation(channels=8, num_horizons=3, propagation_steps=2, alpha=0.1, dropout=0.0, use_refinement=True)

    y_no = no_refine(z_current, a_phys, a_wind_hat, a_sem_hat)
    y_yes = with_refine(z_current, a_phys, a_wind_hat, a_sem_hat)

    assert y_no.shape == (2, 3, 5, 8)
    assert y_yes.shape == (2, 3, 5, 8)
    assert torch.isfinite(y_no).all()
    assert torch.isfinite(y_yes).all()


def test_future_spatial_dependency_sparse_topk():
    torch.manual_seed(0)
    graph_gen = FutureSpatialDependencyGenerator(latent_dim=8, hidden_dim=16, residual_scale=0.1, k=2)
    z_graph = torch.randn(2, 3, 5, 8)
    a_wind_current_dense = torch.rand(2, 5, 5)
    a_sem_current_dense = torch.rand(2, 5, 5)

    a_wind_sparse, a_sem_sparse = graph_gen(
        z_graph,
        a_wind_current_dense,
        a_sem_current_dense,
        return_sparse=True,
    )

    assert isinstance(a_wind_sparse, list) and len(a_wind_sparse) == 3
    assert isinstance(a_sem_sparse, list) and len(a_sem_sparse) == 3

    for horizon_idx in range(3):
        wind_edge_index, wind_edge_weight = a_wind_sparse[horizon_idx]
        sem_edge_index, sem_edge_weight = a_sem_sparse[horizon_idx]

        assert wind_edge_index.shape[0] == 2
        assert sem_edge_index.shape[0] == 2
        assert wind_edge_weight.shape[0] == wind_edge_index.shape[1]
        assert sem_edge_weight.shape[0] == sem_edge_index.shape[1]
        assert wind_edge_index.min().item() >= 0
        assert sem_edge_index.min().item() >= 0
        assert wind_edge_index.max().item() < 5 * 2
        assert sem_edge_index.max().item() < 5 * 2


def test_future_spatial_dependency_dense_keeps_only_k_final_edges_per_row():
    torch.manual_seed(0)
    graph_gen = FutureSpatialDependencyGenerator(latent_dim=4, hidden_dim=8, residual_scale=0.1, k=2, candidate_scale=3)
    z_graph = torch.randn(1, 2, 5, 4)
    a_wind_current_dense = torch.rand(1, 5, 5)
    a_sem_current_dense = torch.rand(1, 5, 5)

    a_wind_hat, _ = graph_gen(z_graph, a_wind_current_dense, a_sem_current_dense, return_sparse=False)

    nonzero_per_row = (a_wind_hat[0, 0] != 0).float().sum(dim=1)
    assert (nonzero_per_row <= 2).all()


def test_future_wind_topk_selects_incoming_sources_and_sparse_edges_match_dense():
    graph_gen = FutureSpatialDependencyGenerator(latent_dim=2, residual_scale=0.0, k=1)
    z_graph = torch.zeros(1, 1, 3, 2)
    incoming = torch.tensor([[[0.0, 0.8, 0.2], [0.1, 0.0, 0.9], [0.7, 0.3, 0.0]]])
    semantic = torch.zeros_like(incoming)

    wind_dense, _ = graph_gen(z_graph, incoming, semantic, return_sparse=False)
    wind_sparse, _ = graph_gen(z_graph, incoming, semantic, return_sparse=True)
    edge_index, edge_weight = wind_sparse[0]

    reconstructed = torch.zeros_like(wind_dense[0, 0])
    reconstructed[edge_index[1], edge_index[0]] = edge_weight
    assert torch.equal(reconstructed, wind_dense[0, 0])
    assert torch.equal(edge_index[0], torch.tensor([1, 2, 0]))
    assert torch.equal(edge_index[1], torch.tensor([0, 1, 2]))


def test_propagator_aggregates_incoming_rows_at_destination():
    class WindOnlyGate(torch.nn.Module):
        def forward(self, hidden, phys, wind, sem, horizon):
            ones = torch.ones_like(hidden)
            zeros = torch.zeros_like(hidden)
            return zeros, ones, zeros

    propagator = MultiGraphAdaptivePropagation(
        channels=1, num_horizons=1, propagation_steps=1, alpha=0.0, dropout=0.0, use_refinement=False
    )
    propagator.gate = WindOnlyGate()
    incoming = torch.zeros(1, 1, 3, 3)
    incoming[0, 0, 1, 0] = 1.0
    hidden = torch.tensor([[[2.0], [5.0], [9.0]]])

    result = propagator(hidden, torch.zeros(3, 3), incoming, torch.zeros_like(incoming))
    assert torch.equal(result[0, 0, :, 0], torch.tensor([0.0, 2.0, 0.0]))


def test_propagation_accepts_sparse_future_graphs():
    torch.manual_seed(0)
    z_current = torch.randn(2, 5, 8)
    a_phys = torch.rand(5, 5)

    graph_gen = FutureSpatialDependencyGenerator(latent_dim=8, hidden_dim=16, residual_scale=0.1, k=2)
    z_graph = torch.randn(2, 3, 5, 8)
    a_wind_current_dense = torch.rand(2, 5, 5)
    a_sem_current_dense = torch.rand(2, 5, 5)
    a_wind_sparse, a_sem_sparse = graph_gen(
        z_graph,
        a_wind_current_dense,
        a_sem_current_dense,
        return_sparse=True,
    )

    propagator = MultiGraphAdaptivePropagation(
        channels=8,
        num_horizons=3,
        propagation_steps=2,
        alpha=0.1,
        dropout=0.0,
        use_refinement=False,
    )
    propagated = propagator(z_current, a_phys, a_wind_sparse, a_sem_sparse)

    assert propagated.shape == (2, 3, 5, 8)
    assert torch.isfinite(propagated).all()


if __name__ == '__main__':
    test_downstream_pipeline_modules()
    test_future_spatial_dependency_sparse_topk()
    test_propagation_accepts_sparse_future_graphs()
    print('downstream pipeline smoke passed')
