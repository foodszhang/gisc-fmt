import torch

from minr_fmt.loss import MorphologyAwareDensityLoss
from minr_fmt.models.ssq_fmt import SSQFMT
from tests.test_ssq_gradients import _has_grad
from tests.test_ssq_math_invariants import make_batch, make_cfg


def test_branch_sdf_composes_with_pi_and_runs_only_when_requested():
    model = SSQFMT(make_cfg(mmax=2, lambda_sdf=0.0))
    batch = make_batch(mmax=2)
    model.eval()
    out_eval = model(
        batch["surface_measurements_packed"], batch["query_coordinates_mm"], batch=batch
    )
    assert "branch_sdf" not in out_eval["aux_outputs"]
    out_diag = model(
        batch["surface_measurements_packed"],
        batch["query_coordinates_mm"],
        batch=batch,
        return_diagnostics=True,
    )
    d = out_diag["diagnostics"]
    expected = (d["pi"][..., None] * d["branch_sdf"]).sum(dim=2)
    expected = expected * d["measurement_supported"][..., None].to(dtype=expected.dtype)
    assert torch.allclose(d["composed_sdf"], expected, atol=1e-6)


def test_sdf_loss_backpropagates_to_candidate_morphology_decoder():
    cfg = make_cfg(mmax=2, lambda_sdf=0.05)
    model = SSQFMT(cfg)
    model.train()
    batch = make_batch(mmax=2)
    out = model(batch["surface_measurements_packed"], batch["query_coordinates_mm"], batch=batch)
    target = torch.rand_like(out["density"])
    sdf_target = torch.rand_like(out["aux_outputs"]["sdf"]) * 2 - 1
    loss = MorphologyAwareDensityLoss(lambda_sdf=0.05)(
        out["density"],
        target,
        out["aux_outputs"],
        sdf_targets=sdf_target,
    )["total_loss"]
    loss.backward()
    assert _has_grad(model.candidate_morphology_decoder)
