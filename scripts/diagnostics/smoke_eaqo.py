"""One-batch EAQO smoke checks for SSQ-FMT Phase A."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hydra
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import train  # noqa: E402
from minr_fmt.datamodule import TrainingDataModule  # noqa: E402


def _compose(exp_name: str, overrides: list[str]):
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                f"exp={exp_name}",
                "trainer.accelerator=cpu",
                "trainer.devices=1",
                "trainer.precision=32",
                "data.dataset_type=fmt_simgen",
                "data.num_queries=128",
                "data.sample_num=128",
                "data.eval_sample_num=128",
                "data.query_sampling.num_queries=128",
                "data.train_max_samples=1",
                "data.val_max_samples=1",
                "data.batch_size=1",
                "data.eval_batch_size=1",
                "data.num_workers=0",
                "data.persistent_workers=false",
                "data.prefetch_factor=1",
                "model.ssq_fmt.representation.query_chunk_size=128",
                "model.ssq_fmt.decoder.query_chunk_size=128",
                "model.ssq_fmt.fusion.query_chunk_size=128",
                *overrides,
            ],
        )
    return cfg


def _first_batch(cfg):
    dm = TrainingDataModule(cfg)
    dm.setup("fit")
    return next(iter(dm.train_dataloader()))


def _assert_phase_a(aux: dict) -> None:
    assert aux.get("final_density_path") == "phase_a_shared_unified_decoder"
    assert aux.get("decoder_ablation") == "shared_only"
    context_scale = aux.get("candidate_context_scale")
    assert torch.is_tensor(context_scale)
    assert float(context_scale.detach().cpu()) == 0.0


def _run_forward(exp_name: str, *, backward: bool, overrides: list[str]) -> dict:
    cfg = _compose(exp_name, overrides)
    module_cls = train._lightning_module_cls(cfg)
    module = module_cls(cfg)
    module.train()
    batch = _first_batch(cfg)
    out = module._call_ssq_model(batch, return_diagnostics=False)
    density = out["density"]
    aux = out.get("aux_outputs", {})
    assert density is not None and density.dim() == 3 and density.shape[-1] == 1
    assert isinstance(aux, dict)
    _assert_phase_a(aux)
    assert "query_view_valid" in aux
    assert "measurement_supported_binary" in aux or not cfg.model.ssq_fmt.eaqo.enabled
    forbidden = {"legacy_shared_logit", "moe_final_density", "phsa_final_density"}
    assert not any(key in aux for key in forbidden)
    if cfg.model.ssq_fmt.eaqo.enabled:
        assert "eaqo_score" in aux
        assert "eaqo_loss_weight" in aux
        assert aux["eaqo_score"].shape == density.squeeze(-1).shape
        assert aux["eaqo_loss_weight"].shape == density.squeeze(-1).shape
        assert "query_loss_weight" in aux
        assert "measurement_supported_binary" in aux
    loss_dict = module.ssq_loss_func(
        density,
        batch["point_densities"].unsqueeze(-1),
        aux,
        gt_voxels=batch.get("gt_voxels"),
        points_ijk=batch.get("points_ijk"),
        query_component_ids=batch.get("query_component_ids"),
        gt_component_centers_mm=batch.get("gt_component_centers_mm"),
        gt_component_valid_mask=batch.get("gt_component_valid_mask"),
    )
    loss = loss_dict["total_loss"]
    assert torch.isfinite(loss.detach())
    if backward:
        loss.backward()
    return aux


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    overrides = list(args.override)

    off_aux = _run_forward("fmt_simgen_v2_ssq_eaqo_off", backward=False, overrides=overrides)
    view_aux = _run_forward("fmt_simgen_v2_ssq_eaqo_view_only", backward=True, overrides=overrides)
    pred_aux = _run_forward("fmt_simgen_v2_ssq_eaqo_pred_only", backward=True, overrides=overrides)

    assert "eaqo_score" not in off_aux
    assert "eaqo/view_ambiguity_mean" in view_aux
    assert torch.isfinite(view_aux["eaqo/view_ambiguity_mean"])
    assert "eaqo/pred_ambiguity_mean" in pred_aux
    print("EAQO smoke checks passed")


if __name__ == "__main__":
    main()
