import sys

import torch

sys.path.insert(0, 'c:/Users/Mohsen/Documents/GHI')

from Src.latent_graph_evolution_encoder import LatentGraphEvolutionEncoder


def test_lgee_shapes_and_finite_values():
    torch.manual_seed(0)
    z_hist = torch.randn(2, 16, 4, 8)
    encoder = LatentGraphEvolutionEncoder(
        latent_dim=8,
        num_horizons=3,
        num_heads=2,
        dropout=0.0,
        ff_hidden_dim=16,
    )

    out = encoder(z_hist)

    assert out.shape == (2, 3, 4, 8)
    assert torch.isfinite(out).all()


if __name__ == '__main__':
    test_lgee_shapes_and_finite_values()
    print('latent graph evolution encoder smoke passed')
