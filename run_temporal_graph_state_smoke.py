import sys
sys.path.insert(0, 'c:/Users/Mohsen/Documents/GHI')
import torch
from Src.temporal_graph_state import TemporalGraphStateAggregator, LatentGraphGenerator

B, W, N, C = 2, 6, 4, 8
Z = torch.randn(B, W, N, C)
S = TemporalGraphStateAggregator(channels=C)(Z)
print('S', S.shape)
edge_index = torch.tensor([[0,0,0,1,1,2],[1,2,3,2,3,3]])
hist_adj = torch.rand(B, N, N)
gen = LatentGraphGenerator(state_dim=C, hidden_dim=16, n_horizons=3, horizon_emb_dim=C)
out = gen(hist_adj, S, edge_index)
print('out', out.shape)
