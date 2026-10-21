import ast
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Src.LGEE import LatentGraphEvolutionEncoder
from Src.data_structure import TemporalFold, build_windowed_folds


def test_lgee_shapes_and_legacy_shared_query_path():
    torch.manual_seed(0)
    z_hist = torch.randn(2, 5, 3, 8)
    cov_fut = torch.randn(2, 4, 3, 5)
    z_state = torch.randn(2, 3, 8)
    encoder = LatentGraphEvolutionEncoder(8, 4, dropout=0.0, cov_dim=5)

    assert encoder(z_hist).shape == (2, 4, 3, 8)
    assert encoder(z_hist, cov_fut).shape == (2, 4, 3, 8)
    assert encoder(z_hist, None, z_state).shape == (2, 4, 3, 8)
    assert encoder(z_hist, cov_fut, z_state).shape == (2, 4, 3, 8)

    legacy = LatentGraphEvolutionEncoder(8, 4, dropout=0.0, cov_dim=None)
    z_flat, queries, (batch_size, num_nodes) = legacy._prepare_inputs(z_hist)
    for block in legacy.blocks:
        queries = block(queries, z_flat, z_flat)
    expected = legacy._reshape_output(queries, batch_size, num_nodes)
    assert torch.equal(legacy(z_hist), expected)


def test_lgee_queries_respond_to_future_covariates_and_current_state():
    torch.manual_seed(1)
    encoder = LatentGraphEvolutionEncoder(8, 3, dropout=0.0, cov_dim=5).eval()
    z_hist = torch.randn(2, 5, 3, 8)
    cov_fut = torch.randn(2, 3, 3, 5)
    z_state = torch.randn(2, 3, 8)

    baseline = encoder(z_hist, cov_fut, z_state)
    changed_cov = encoder(z_hist, cov_fut + 1.0, z_state)
    changed_state = encoder(z_hist, cov_fut, z_state + 1.0)

    assert not torch.allclose(baseline, changed_cov)
    assert not torch.allclose(baseline, changed_state)


def test_covariate_names_exclude_weather_and_target_features():
    notebook = json.loads((REPO_ROOT / "NoteBooks" / "pipeline.ipynb").read_text(encoding="utf-8"))
    source = "".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assignment = next(line for line in source.splitlines() if "covariate_names =" in line)
    covariate_names = ast.literal_eval(assignment.split("=", 1)[1].strip())

    assert covariate_names == ("tod_sin", "tod_cos", "sun_elev_sin", "sun_elev_cos", "toa_norm")
    assert not any(name.startswith("wind_") or name in {"tcc", "GHI"} for name in covariate_names)


def test_cov_future_alignment_uses_t_plus_h():
    time_steps, nodes, features = 12, 2, 5
    tensor = np.broadcast_to(
        np.arange(time_steps, dtype=np.float32)[:, None, None],
        (time_steps, nodes, features),
    )
    fold = TemporalFold("toy", (0, time_steps), None, (0, time_steps))
    cov_mask = np.array([False, True, False, True, True])
    windowed = build_windowed_folds(
        tensor,
        {},
        [fold],
        T=3,
        H=2,
        cov_mask=cov_mask,
    )[0].train

    assert windowed.cov_future.shape == (8, 2, nodes, 3)
    for h in range(2):
        assert np.all(windowed.cov_future[0, h] == 3 + h)


if __name__ == "__main__":
    test_lgee_shapes_and_legacy_shared_query_path()
    test_lgee_queries_respond_to_future_covariates_and_current_state()
    test_covariate_names_exclude_weather_and_target_features()
    test_cov_future_alignment_uses_t_plus_h()
    print("LGEE sample query tests passed")
