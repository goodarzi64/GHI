import sys

import torch

sys.path.insert(0, 'c:/Users/Mohsen/Documents/GHI')

from Src.forecast_head import ForecastAndGraphLoss, ForecastHead
from Src.future_spatial_dependency import FutureSpatialDependencyGenerator
from Src.future_spatial_propagation import MultiGraphAdaptivePropagation
from Src.temporal_conv_module import TemporalContextEncoder


def main():
    torch.manual_seed(0)

    B, T, N, F = 2, 24, 5, 8
    H = 3
    C = 8

    x_seq = torch.randn(B, T, N, F)
    a_phys = torch.rand(N, N)
    a_phys = (a_phys + a_phys.T) / 2.0
    a_phys.fill_diagonal_(0.0)
    a_wind_current_dense = torch.rand(B, N, N)
    a_sem_current_dense = torch.rand(B, N, N)
    target = torch.rand(B, H, N)
    wind_target = a_wind_current_dense.unsqueeze(1).expand(-1, H, -1, -1)
    sem_target = a_sem_current_dense.unsqueeze(1).expand(-1, H, -1, -1)

    temporal_encoder = TemporalContextEncoder(
        in_channels=F,
        out_channels=C,
        num_blocks=3,
        kernel_size=3,
        dilation_list=[1, 2, 4],
        dropout=0.0,
    )
    z_hist = temporal_encoder(x_seq)
    z_current = z_hist[:, -1, :, :]
    z_graph = z_current.unsqueeze(1).expand(-1, H, -1, -1)

    graph_gen = FutureSpatialDependencyGenerator(latent_dim=C, hidden_dim=16, residual_scale=0.1, k=2)
    a_wind_hat, a_sem_hat = graph_gen(z_graph, a_wind_current_dense, a_sem_current_dense, return_sparse=False)

    propagator = MultiGraphAdaptivePropagation(
        channels=C,
        num_horizons=H,
        propagation_steps=2,
        alpha=0.1,
        dropout=0.0,
        use_refinement=False,
    )
    h_future = propagator(z_current, a_phys, a_wind_hat, a_sem_hat)

    forecast_head = ForecastHead(channels=C, dropout=0.0)
    forecast = forecast_head(h_future)

    loss_fn = ForecastAndGraphLoss(forecast_weight=1.0, graph_weight=0.1)
    loss = loss_fn(forecast, target, a_wind_hat, wind_target, a_sem_hat, sem_target)

    print('Input shape           :', tuple(x_seq.shape))
    print('Z_hist shape          :', tuple(z_hist.shape))
    print('Z_graph shape         :', tuple(z_graph.shape))
    print('Future wind graph     :', tuple(a_wind_hat.shape))
    print('Future semantic graph :', tuple(a_sem_hat.shape))
    print('FSDP output shape     :', tuple(h_future.shape))
    print('Forecast shape        :', tuple(forecast.shape))
    print('Loss value            :', float(loss.detach().cpu()))

    assert z_hist.shape == (B, 3, N, C)
    assert z_graph.shape == (B, H, N, C)
    assert a_wind_hat.shape == (B, H, N, N)
    assert a_sem_hat.shape == (B, H, N, N)
    assert h_future.shape == (B, H, N, C)
    assert forecast.shape == (B, H, N)
    assert torch.isfinite(loss)


if __name__ == '__main__':
    main()
    print('end-to-end forward pass smoke passed')
