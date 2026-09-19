import sys

import torch

sys.path.insert(0, 'c:/Users/Mohsen/Documents/GHI')

from Src.future_spatial_dependency import FutureSpatialDependencyGenerator

B, H, N, C = 2, 3, 5, 8
z_graph = torch.randn(B, H, N, C)
a_wind_current = torch.rand(B, N, N)
a_sem_current = torch.rand(B, N, N)

graph_gen = FutureSpatialDependencyGenerator(latent_dim=C, hidden_dim=16, residual_scale=0.1)
a_wind_hat, a_sem_hat = graph_gen(z_graph, a_wind_current, a_sem_current)

print('a_wind_hat', a_wind_hat.shape)
print('a_sem_hat', a_sem_hat.shape)
