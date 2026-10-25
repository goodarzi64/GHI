import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Src.FSDG import FutureSpatialDependencyGenerator


def _inputs(batch_size: int = 2, horizons: int = 3, nodes: int = 4, latent_dim: int = 5):
    torch.manual_seed(13)
    z_graph = torch.randn(batch_size, horizons, nodes, latent_dim)
    a_wind = torch.rand(batch_size, nodes, nodes)
    a_semantic = torch.rand(batch_size, nodes, nodes)
    return z_graph, a_wind, a_semantic


def _persistence(current: torch.Tensor, symmetric: bool = False) -> torch.Tensor:
    nodes = current.shape[-1]
    diagonal = torch.eye(nodes, dtype=torch.bool, device=current.device)
    clipped = current.clamp(0.0, 1.0).masked_fill(diagonal, 0.0)
    if symmetric:
        clipped = torch.maximum(clipped, clipped.transpose(-1, -2))
    return clipped


def test_zero_initializes_last_layers_and_matches_persistence():
    z_graph, a_wind, a_semantic = _inputs()
    generator = FutureSpatialDependencyGenerator(latent_dim=5, zero_init_output=True)

    for mlp in (generator.wind_mlp, generator.sem_mlp):
        assert torch.count_nonzero(mlp[-1].weight).item() == 0
        assert torch.count_nonzero(mlp[-1].bias).item() == 0

    wind_hat, semantic_hat = generator(z_graph, a_wind, a_semantic)
    expected_wind = _persistence(a_wind).unsqueeze(1).expand_as(wind_hat)
    expected_semantic = _persistence(a_semantic, symmetric=True).unsqueeze(1).expand_as(semantic_hat)
    assert torch.allclose(wind_hat, expected_wind, atol=1e-7, rtol=0.0)
    assert torch.allclose(semantic_hat, expected_semantic, atol=1e-7, rtol=0.0)


def test_default_false_keeps_standard_output_initialization():
    z_graph, a_wind, a_semantic = _inputs()
    generator = FutureSpatialDependencyGenerator(latent_dim=5, zero_init_output=False)
    wind_hat, _ = generator(z_graph, a_wind, a_semantic)
    expected_wind = _persistence(a_wind).unsqueeze(1).expand_as(wind_hat)
    assert not torch.allclose(wind_hat, expected_wind, atol=1e-7, rtol=0.0)


def test_output_ranges_and_gradients_after_initial_update():
    z_graph, a_wind, a_semantic = _inputs()
    generator = FutureSpatialDependencyGenerator(latent_dim=5, zero_init_output=True)
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-2)
    target = _persistence(a_wind).unsqueeze(1).expand(-1, z_graph.shape[1], -1, -1).clone()
    target[..., 0, 1] = 0.0  # Ensure the target differs from persistence for this edge.

    wind_hat, semantic_hat = generator(z_graph, a_wind, a_semantic)
    assert wind_hat.shape == (2, 3, 4, 4)
    assert semantic_hat.shape == (2, 3, 4, 4)
    for output in (wind_hat, semantic_hat):
        assert torch.isfinite(output).all()
        assert torch.all((output >= 0.0) & (output <= 1.0))

    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(wind_hat, target)
    loss.backward()
    optimizer.step()
    assert torch.count_nonzero(generator.wind_mlp[-1].weight).item() > 0

    optimizer.zero_grad(set_to_none=True)
    wind_hat, _ = generator(z_graph, a_wind, a_semantic)
    torch.nn.functional.mse_loss(wind_hat, target).backward()
    for parameter in generator.wind_mlp[0].parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


if __name__ == "__main__":
    test_zero_initializes_last_layers_and_matches_persistence()
    test_default_false_keeps_standard_output_initialization()
    test_output_ranges_and_gradients_after_initial_update()
    print("FSDG zero initialization tests passed")
