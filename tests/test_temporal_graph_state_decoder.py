import sys

import torch

sys.path.insert(0, 'c:/Users/Mohsen/Documents/GHI')

from Src.temporal_graph_state import HorizonAwareMultiGraphAPPNP, ForecastHead, ForecastAndGraphLoss


def test_horizon_aware_multi_graph_appnp_shapes():
    torch.manual_seed(0)
    model = HorizonAwareMultiGraphAPPNP(channels=8, horizon_emb_dim=4, propagation_steps=3, alpha=0.1)

    z = torch.randn(2, 5, 8)
    adj_phys = torch.rand(2, 5, 5)
    adj_wind = torch.rand(2, 5, 5)
    adj_sem = torch.rand(2, 5, 5)

    out = model(z, adj_phys, adj_wind, adj_sem, horizon_idx=1)

    assert out.shape == z.shape
    assert torch.isfinite(out).all()


def test_forecast_head_and_loss():
    torch.manual_seed(0)
    head = ForecastHead(channels=8, num_horizons=3, horizon_emb_dim=4)
    state = torch.randn(2, 5, 8)
    preds = head(state, horizon_idx=torch.tensor([0, 1]))

    assert preds.shape == (2, 3, 5)
    assert torch.isfinite(preds).all()

    loss_fn = ForecastAndGraphLoss(forecast_weight=1.0, graph_weight=0.5, sparsity_weight=0.01)
    target = torch.randn_like(preds)
    wind_graph = torch.rand(2, 3, 5, 5)
    sem_graph = torch.rand(2, 3, 5, 5)
    wind_target = torch.rand_like(wind_graph)
    sem_target = torch.rand_like(sem_graph)

    loss = loss_fn(preds, target, wind_graph, wind_target, sem_graph, sem_target)

    assert torch.isfinite(loss)
    assert loss >= 0


if __name__ == '__main__':
    test_horizon_aware_multi_graph_appnp_shapes()
    test_forecast_head_and_loss()
    print('decoder propagation smoke passed')
