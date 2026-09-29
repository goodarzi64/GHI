import sys

import torch

sys.path.insert(0, 'c:/Users/Mohsen/Documents/GHI')

from Src.forecast_head import ForecastAndGraphLoss, ForecastHead
from Src.future_spatial_dependency import FutureSpatialDependencyGenerator
from Src.future_spatial_propagation import MultiGraphAdaptivePropagation
from Src.temporal_conv_module import TemporalContextEncoder


class EndToEndForwardSmokeModel(torch.nn.Module):
    """Minimal end-to-end smoke wrapper for the currently maintained forward path."""

    def __init__(self, *, batch_size: int = 2, time_steps: int = 24, num_nodes: int = 5, in_features: int = 8, hidden_dim: int = 8, num_horizons: int = 3):
        super().__init__()
        self.batch_size = batch_size
        self.num_horizons = num_horizons
        self.temporal_encoder = TemporalContextEncoder(
            in_channels=in_features,
            out_channels=hidden_dim,
            num_blocks=3,
            kernel_size=3,
            dilation_list=[1, 2, 4],
            dropout=0.0,
        )
        self.graph_generator = FutureSpatialDependencyGenerator(
            latent_dim=hidden_dim,
            hidden_dim=2 * hidden_dim,
            residual_scale=0.1,
            k=2,
            candidate_scale=3,
        )
        self.propagator = MultiGraphAdaptivePropagation(
            channels=hidden_dim,
            num_horizons=num_horizons,
            propagation_steps=2,
            alpha=0.1,
            dropout=0.0,
            use_refinement=False,
        )
        self.forecast_head = ForecastHead(channels=hidden_dim, dropout=0.0)
        self.loss_fn = ForecastAndGraphLoss(forecast_weight=1.0, graph_weight=0.1)

    def forward(self, x_seq: torch.Tensor, a_phys: torch.Tensor, a_wind_current: torch.Tensor, a_sem_current: torch.Tensor, target: torch.Tensor):
        z_hist = self.temporal_encoder(x_seq)
        z_current = z_hist[:, -1, :, :]
        z_graph = z_current.unsqueeze(1).expand(-1, self.num_horizons, -1, -1)

        a_wind_hat, a_sem_hat = self.graph_generator(
            z_graph,
            a_wind_current,
            a_sem_current,
            return_sparse=False,
        )

        h_future = self.propagator(z_current, a_phys, a_wind_hat, a_sem_hat)
        forecast = self.forecast_head(h_future)
        loss = self.loss_fn(
            forecast,
            target,
            a_wind_hat,
            a_wind_current.unsqueeze(1).expand(-1, self.num_horizons, -1, -1),
            a_sem_hat,
            a_sem_current.unsqueeze(1).expand(-1, self.num_horizons, -1, -1),
        )

        return {
            'x_seq': x_seq,
            'z_hist': z_hist,
            'z_current': z_current,
            'z_graph': z_graph,
            'a_wind_hat': a_wind_hat,
            'a_sem_hat': a_sem_hat,
            'h_future': h_future,
            'forecast': forecast,
            'loss': loss,
        }


def test_end_to_end_forward_smoke():
    torch.manual_seed(0)

    B, T, N, F = 2, 24, 5, 8
    H = 3
    x_seq = torch.randn(B, T, N, F)
    a_phys = torch.rand(N, N)
    a_phys = (a_phys + a_phys.T) / 2.0
    a_phys = a_phys.fill_diagonal_(0.0)
    a_wind_current = torch.rand(B, N, N)
    a_sem_current = torch.rand(B, N, N)
    target = torch.rand(B, H, N)
    wind_target = a_wind_current.unsqueeze(1).expand(-1, H, -1, -1)
    sem_target = a_sem_current.unsqueeze(1).expand(-1, H, -1, -1)

    model = EndToEndForwardSmokeModel(batch_size=B, time_steps=T, num_nodes=N, in_features=F, hidden_dim=8, num_horizons=H)
    out = model(x_seq, a_phys, a_wind_current, a_sem_current, target)

    assert out['x_seq'].shape == (B, T, N, F)
    assert out['z_hist'].shape[0] == B
    assert out['z_hist'].shape[2] == N
    assert out['z_hist'].shape[3] == 8
    assert out['z_graph'].shape == (B, H, N, 8)
    assert out['a_wind_hat'].shape == (B, H, N, N)
    assert out['a_sem_hat'].shape == (B, H, N, N)
    assert out['h_future'].shape == (B, H, N, 8)
    assert out['forecast'].shape == (B, H, N)
    assert torch.isfinite(out['forecast']).all()
    assert torch.isfinite(out['loss'])


def test_temporal_encoder_and_graph_generator_still_have_valid_interface():
    torch.manual_seed(0)

    x_seq = torch.randn(2, 24, 5, 8)
    encoder = TemporalContextEncoder(in_channels=8, out_channels=8, num_blocks=3, kernel_size=3, dilation_list=[1, 2, 4], dropout=0.0)
    z_hist = encoder(x_seq)
    assert z_hist.shape[0] == 2
    assert z_hist.shape[2] == 5
    assert z_hist.shape[3] == 8

    z_current = z_hist[:, -1, :, :]
    z_graph = z_current.unsqueeze(1).expand(-1, 3, -1, -1)
    graph_gen = FutureSpatialDependencyGenerator(latent_dim=8, hidden_dim=16, residual_scale=0.1, k=2)
    a_wind_hat, a_sem_hat = graph_gen(z_graph, torch.rand(2, 5, 5), torch.rand(2, 5, 5), return_sparse=False)

    assert z_graph.shape == (2, 3, 5, 8)
    assert a_wind_hat.shape == (2, 3, 5, 5)
    assert a_sem_hat.shape == (2, 3, 5, 5)


if __name__ == '__main__':
    test_end_to_end_forward_smoke()
    test_temporal_encoder_and_graph_generator_still_have_valid_interface()
    print('end-to-end pipeline smoke passed')
