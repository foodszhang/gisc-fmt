import torch

from minr_fmt.models.ssq_fmt import SSQFMT
from tests.test_ssq_math_invariants import make_batch, make_cfg


def test_footprint_mask_normalizes_only_valid_samples():
    cfg = make_cfg(mmax=1)
    cfg.model.ssq_fmt.local_sampling.template = "grid3x3"
    cfg.model.ssq_fmt.local_sampling.offsets_px = None
    model = SSQFMT(cfg)
    batch = make_batch(mmax=1)
    batch["detector_valid_mask"][:, :, :8, :] = False
    out = model(
        batch["surface_measurements_packed"],
        batch["query_coordinates_mm"],
        batch=batch,
        return_diagnostics=True,
    )
    d = out["diagnostics"]
    sums = d["A"].sum(dim=-1)
    valid_qv = d["sample_valid"].any(dim=-1)
    assert torch.allclose(sums[valid_qv], torch.ones_like(sums[valid_qv]), atol=1e-5)
    assert torch.all(d["A"][~d["sample_valid"]] == 0)
