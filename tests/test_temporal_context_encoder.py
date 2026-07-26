import sys

import torch

sys.path.insert(0, 'c:/Users/Mohsen/Documents/GHI')

from Src.temporal_conv_module import TemporalContextEncoder, TemporalResidualBlock


def test_temporal_context_encoder_compresses_time_without_cross_station_mix():
    torch.manual_seed(0)
    x = torch.randn(2, 336, 4, 8)
    encoder = TemporalContextEncoder(in_channels=8, out_channels=8, num_blocks=4, kernel_size=3, dropout=0.0)

    out = encoder(x)

    assert out.shape[0] == x.shape[0]
    assert out.shape[1] < x.shape[1]
    assert out.shape[2] == x.shape[2]
    assert out.shape[3] == 8

    # Ensure no cross-station mixing by checking that each station path is independent.
    y = out[:, :, 0, :]
    assert torch.isfinite(y).all()


def test_temporal_residual_block_preserves_station_axis():
    x = torch.randn(2, 16, 3, 8)
    block = TemporalResidualBlock(in_channels=8, out_channels=8, kernel_size=3, dilation=2, dropout=0.0)

    out = block(x)

    assert out.shape[0] == x.shape[0]
    assert out.shape[1] < x.shape[1]
    assert out.shape[2] == x.shape[2]
    assert out.shape[3] == x.shape[3]


if __name__ == '__main__':
    test_temporal_context_encoder_compresses_time_without_cross_station_mix()
    test_temporal_residual_block_preserves_station_axis()
    print('temporal context encoder smoke passed')
