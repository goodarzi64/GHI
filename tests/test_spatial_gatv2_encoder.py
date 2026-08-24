import sys

import torch

sys.path.insert(0, 'c:/Users/Mohsen/Documents/GHI')

from Src.spatial_gatv2_encoder import SpatialGATv2EncoderBatched


def test_shared_multigraph_gatv2_has_single_encoder_and_graph_specific_biases():
    torch.manual_seed(0)
    x_batch = torch.randn(2, 5, 8)
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        [1, 2, 0, 2, 0, 1, 1, 2, 0, 1],
    ])
    adj = torch.rand(2, 5, 5)
    adj = (adj + adj.transpose(-1, -2)) / 2.0
    adj = adj.clamp_min(1e-6)

    encoder = SpatialGATv2EncoderBatched(in_features=8, out_features=16, heads=2, dropout=0.0)

    assert hasattr(encoder, 'gat')
    assert hasattr(encoder, 'graph_betas')
    assert encoder.gat.lin_l is encoder.gat.lin_r
    assert encoder.graph_betas.shape == (3,)

    out = encoder(x_batch, [edge_index, edge_index, edge_index], [adj, adj, adj])

    assert len(out) == 3
    assert all(o.shape == (2, 5, 16 * 2) for o in out)
    assert torch.isfinite(torch.stack(out, dim=0)).all()


if __name__ == '__main__':
    test_shared_multigraph_gatv2_has_single_encoder_and_graph_specific_biases()
    print('spatial GATv2 shared-encoder smoke passed')
