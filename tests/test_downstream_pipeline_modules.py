import sys

import torch

sys.path.insert(0, 'c:/Users/Mohsen/Documents/GHI')

from Src.forecast_head import ForecastHead
from Src.future_spatial_dependency import FutureSpatialDependencyGenerator
from Src.future_spatial_propagation import CurrentStateRefinement, MultiGraphAdaptivePropagation


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

    propagator = MultiGraphAdaptivePropagation(channels=8, num_horizons=3, propagation_steps=2, alpha=0.1, dropout=0.0)
    a_phys = torch.rand(5, 5)
    propagated = propagator(z_refined, a_phys, a_wind_hat, a_sem_hat)
    assert propagated.shape == (2, 3, 5, 8)
    assert torch.isfinite(propagated).all()

    head = ForecastHead(channels=8, dropout=0.0)
    forecast = head(propagated)
    assert forecast.shape == (2, 3, 5)
    assert torch.isfinite(forecast).all()


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


if __name__ == '__main__':
    test_downstream_pipeline_modules()
    test_future_spatial_dependency_sparse_topk()
    print('downstream pipeline smoke passed')
