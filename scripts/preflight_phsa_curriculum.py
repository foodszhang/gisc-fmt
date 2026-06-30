#!/usr/bin/env python3
"""Fail-fast audit for the 2400-sample PHSA curriculum.

This script checks dataset isolation, the instantiated Patch-2 aggregation semantics,
optimizer/freezing groups, normalization ownership, stage/checkpoint cadence and a
small differentiable forward/backward before a long run is allowed to start.
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from torch.utils.data._utils.collate import default_collate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.datamodule import TrainingDataModule  # noqa: E402
from minr_fmt.module import TrainingLightningModule  # noqa: E402
from minr_fmt.phsa_sample_level import (  # noqa: E402
    _PassthroughCandidateNormalizer,
    activate_phsa_sample_level_hypotheses,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, required=True)
    parser.add_argument("--val-samples", type=int, required=True)
    parser.add_argument("--test-samples", type=int, default=300)
    parser.add_argument("--train-queries", type=int, required=True)
    parser.add_argument("--eval-queries", type=int, required=True)
    parser.add_argument("--hypothesis-points", type=int, required=True)
    parser.add_argument("--hypothesis-seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--phase-a-epochs", type=int, required=True)
    parser.add_argument("--phase-b-epochs", type=int, required=True)
    parser.add_argument("--full-epochs", type=int, required=True)
    parser.add_argument("--val-every", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-forward-smoke", action="store_true")
    return parser.parse_args()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def common_overrides(args: argparse.Namespace, *, smoke: bool = False) -> list[str]:
    train_queries = min(args.train_queries, 256) if smoke else args.train_queries
    eval_queries = min(args.eval_queries, 256) if smoke else args.eval_queries
    hypothesis_points = min(args.hypothesis_points, 256) if smoke else args.hypothesis_points
    train_samples = 1 if smoke else args.train_samples
    val_samples = 1 if smoke else args.val_samples
    output = Path(tempfile.gettempdir()) / "phsa_preflight"
    return [
        "model=ssq_fmt",
        "exp=fmt_simgen_v2_view_complementary",
        "data.dataset_type=fmt_simgen",
        f"seed={args.seed}",
        f"data.train_max_samples={train_samples}",
        f"data.val_max_samples={val_samples}",
        "data.subset_policy=random",
        f"data.subset_seed={args.seed}",
        "data.projection_norm=raw",
        f"data.sample_num={train_queries}",
        f"data.num_queries={train_queries}",
        f"data.query_sampling.num_queries={train_queries}",
        f"data.eval_sample_num={eval_queries}",
        "data.num_workers=0",
        f"data.batch_size={1 if smoke else args.batch_size}",
        "data.eval_batch_size=1",
        "data.pin_memory=false",
        "data.persistent_workers=false",
        "data.prefetch_factor=1",
        "data.resample_queries_each_epoch=true",
        "data.descatter_target_files=[]",
        "++data.load_stage1_prior=false",
        "++data.load_stage1_mesh=false",
        "model.ssq_fmt.view_complementary.ablation=full",
        "model.ssq_fmt.view_complementary.separability_mode=geometry_measurement",
        "++model.ssq_fmt.view_complementary.strong_shared_fusion=true",
        "++model.ssq_fmt.view_complementary.decoder_fusion_mode=joint_nonresidual",
        "++model.ssq_fmt.view_complementary.sample_level_hypotheses.enabled=true",
        f"++model.ssq_fmt.view_complementary.sample_level_hypotheses.count={hypothesis_points}",
        f"++model.ssq_fmt.view_complementary.sample_level_hypotheses.seed={args.hypothesis_seed}",
        "++model.ssq_fmt.view_complementary.hypothesis_grid.enabled=false",
        "++model.ssq_fmt.view_complementary.routing.enabled=false",
        "++model.ssq_fmt.view_complementary.continuous_applicability=true",
        "++model.ssq_fmt.view_complementary.candidate_hidden_injection=true",
        "model.ssq_fmt.view_complementary.lambda_separability_measurement=0.05",
        "++model.ssq_fmt.memory.checkpoint_encoder=true",
        f"paths.output_dir={output}",
    ]


def make_cfg(
    args: argparse.Namespace,
    phase: str,
    *,
    smoke: bool = False,
) -> Any:
    overrides = common_overrides(args, smoke=smoke)
    if phase == "phase_a":
        overrides.extend(
            [
                "model.ssq_fmt.view_complementary.training_phase=phase_a",
                "++model.ssq_fmt.view_complementary.context_warmup_enabled=true",
                "optim.lr=0.0003",
                "++optim.scheduler.warmup_epochs=5",
                "++optim.scheduler.warmup_start_factor=0.2",
                f"trainer.max_epochs={args.phase_a_epochs}",
            ]
        )
    elif phase == "phase_b":
        overrides.extend(
            [
                "model.ssq_fmt.view_complementary.training_phase=phase_b",
                "++model.ssq_fmt.view_complementary.context_warmup_enabled=true",
                "model.ssq_fmt.view_complementary.context_warmup_steps=500",
                "model.ssq_fmt.view_complementary.lr.encoder=0.0",
                "model.ssq_fmt.view_complementary.lr.constructor=0.00001",
                "model.ssq_fmt.view_complementary.lr.candidate_encoder=0.00003",
                "model.ssq_fmt.view_complementary.lr.candidate_context=0.00003",
                "model.ssq_fmt.view_complementary.lr.decoder=0.00003",
                "model.ssq_fmt.view_complementary.lr.separability=0.00001",
                "model.ssq_fmt.view_complementary.lr.shared=0.000005",
                f"trainer.max_epochs={args.phase_b_epochs}",
            ]
        )
    elif phase == "full":
        overrides.extend(
            [
                "model.ssq_fmt.view_complementary.training_phase=full",
                "++model.ssq_fmt.view_complementary.context_warmup_enabled=false",
                "model.ssq_fmt.view_complementary.lr.encoder=0.00002",
                "model.ssq_fmt.view_complementary.lr.constructor=0.00002",
                "model.ssq_fmt.view_complementary.lr.candidate_encoder=0.00003",
                "model.ssq_fmt.view_complementary.lr.candidate_context=0.00003",
                "model.ssq_fmt.view_complementary.lr.decoder=0.00003",
                "model.ssq_fmt.view_complementary.lr.separability=0.00001",
                "model.ssq_fmt.view_complementary.lr.shared=0.00002",
                f"trainer.max_epochs={args.full_epochs}",
            ]
        )
    else:
        raise ValueError(phase)

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def parameter_group_map(module: TrainingLightningModule) -> dict[int, tuple[str, float]]:
    configured = module.configure_optimizers()
    optimizer = configured["optimizer"] if isinstance(configured, dict) else configured
    result: dict[int, tuple[str, float]] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group_{index}"))
        lr = float(group["lr"])
        for parameter in group["params"]:
            pid = id(parameter)
            check(pid not in result, f"parameter occurs in multiple optimizer groups: {name}")
            result[pid] = (name, lr)
    expected = {id(p) for p in module.parameters() if p.requires_grad}
    check(set(result) == expected, "optimizer does not cover each trainable parameter exactly once")
    return result


def assert_lr(
    mapping: dict[int, tuple[str, float]],
    parameters: Any,
    expected: float,
    label: str,
) -> None:
    values = {mapping[id(parameter)][1] for parameter in parameters if parameter.requires_grad}
    check(values, f"{label} has no trainable parameters")
    check(
        all(math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12) for value in values),
        f"{label} learning rates {sorted(values)} != {expected}",
    )


def audit_splits(cfg: Any, args: argparse.Namespace) -> None:
    dm = TrainingDataModule(cfg)
    dm.setup(stage=None)
    train_ids = {path.name for path in dm.train_dataset.dirs}
    val_ids = {path.name for path in dm.val_dataset.dirs}
    test_ids = {path.name for path in dm.test_dataset.dirs}
    check(len(train_ids) == args.train_samples, f"actual train split={len(train_ids)}, requested={args.train_samples}")
    check(len(val_ids) == args.val_samples, f"actual val split={len(val_ids)}, requested={args.val_samples}")
    check(len(test_ids) == args.test_samples, f"actual test split={len(test_ids)}, expected={args.test_samples}")
    check(train_ids.isdisjoint(val_ids), "train and validation sample IDs overlap")
    check(train_ids.isdisjoint(test_ids), "train and test sample IDs overlap")
    check(val_ids.isdisjoint(test_ids), "validation and test sample IDs overlap")
    print(f"[PASS] splits: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}; disjoint")


def audit_aggregation_and_full_optimizer(cfg: Any, args: argparse.Namespace) -> None:
    module = TrainingLightningModule(cfg)
    net = module.net
    check(net.__class__.__name__ == "SSQFMTPatch2", f"unexpected model class: {net.__class__.__name__}")
    check(bool(net.phsa_sample_level_hypotheses_enabled), "sample-level hypotheses are not enabled")
    check(net.phsa_sample_level_hypothesis_count == args.hypothesis_points, "hypothesis count mismatch")
    check(net.phsa_sample_level_hypothesis_seed == args.hypothesis_seed, "hypothesis seed mismatch")
    check(isinstance(net.candidate_normalizer, _PassthroughCandidateNormalizer), "unused candidate quantile normalizer is still active")
    check(str(cfg.data.projection_norm) == "raw", "dataset must provide raw measurements")
    check(str(net.normalizer.mode) == "per_view_max", "network normalization policy changed unexpectedly")

    aggregation = net.complementary_aggregation
    check(hasattr(aggregation, "common_projection"), "Patch-2 common projection is missing")
    check(aggregation.candidate_projection[0].in_features == 2, "support/separability descriptor input is not [support, separability]")

    torch.manual_seed(7)
    b, v, m = 1, 3, 2
    feature_dim = int(aggregation.feature_norm.normalized_shape[0])
    shared = torch.randn(b, v, feature_dim)
    candidate = torch.randn(b, v, m, feature_dim)
    view_valid = torch.ones(b, v, dtype=torch.bool)
    candidate_valid = torch.ones(b, m, dtype=torch.bool)
    candidate_view_valid = torch.ones(b, v, m, dtype=torch.bool)
    separability = torch.tensor([[[0.2, 0.8], [0.5, 0.3], [0.9, 0.4]]])
    support_a = torch.full((b, v, m), 0.1)
    support_b = torch.tensor([[[0.9, 0.1], [0.2, 0.8], [0.5, 0.4]]])
    out_a = aggregation(
        shared,
        candidate,
        view_valid,
        support_a,
        separability,
        candidate_valid,
        candidate_view_valid=candidate_view_valid,
        geometry_only=True,
    )
    out_b = aggregation(
        shared,
        candidate,
        view_valid,
        support_b,
        separability,
        candidate_valid,
        candidate_view_valid=candidate_view_valid,
        geometry_only=True,
    )
    expected = aggregation.epsilon_s + separability
    expected = expected / expected.sum(dim=1, keepdim=True)
    check(torch.allclose(out_a["view_weights"], expected, atol=1e-7), "PHSA weights do not equal normalized epsilon+separability")
    check(torch.allclose(out_a["view_weights"], out_b["view_weights"], atol=1e-7), "support still directly changes geometry-only view weights")
    descriptor_delta = float((out_a["candidate"] - out_b["candidate"]).abs().max())
    check(descriptor_delta > 1e-7, "support descriptor no longer affects candidate representation")

    mapping = parameter_group_map(module)
    check(aggregation.strong_shared_fusion, "strong query-wise shared fusion is disabled")
    check(net.unified_density_decoder.fusion_mode == "joint_nonresidual", "joint non-residual decoder is disabled")
    assert_lr(mapping, net.complementary_aggregation.common_projection.parameters(), 3e-5, "full common projection")
    assert_lr(mapping, net.complementary_aggregation.candidate_projection.parameters(), 3e-5, "full candidate descriptor projection")
    assert_lr(mapping, net.unified_density_decoder.candidate_context.parameters(), 3e-5, "full candidate context")
    assert_lr(mapping, net.unified_density_decoder.candidate_input.parameters(), 3e-5, "full candidate input")
    assert_lr(mapping, net.unified_density_decoder.head.parameters(), 3e-5, "full density head")
    assert_lr(mapping, net.unified_density_decoder.candidate_norm.parameters(), 2e-5, "full candidate norm")

    names = {id(parameter): name for name, parameter in module.named_parameters()}
    zero_lr_names = [names[pid] for pid, (_group, lr) in mapping.items() if lr == 0.0]
    check(not zero_lr_names, f"unexpected zero-LR trainable parameters: {zero_lr_names[:20]}")
    print("[PASS] instantiated Patch-2 PHSA semantics and full-stage optimizer groups")
    del module
    gc.collect()


def audit_phase_b_optimizer(cfg: Any) -> None:
    module = TrainingLightningModule(cfg)
    net = module.net
    mapping = parameter_group_map(module)
    assert_lr(mapping, net.complementary_aggregation.common_projection.parameters(), 3e-5, "Phase B common projection")
    assert_lr(mapping, net.complementary_aggregation.candidate_projection.parameters(), 3e-5, "Phase B candidate descriptor projection")
    assert_lr(mapping, net.unified_density_decoder.candidate_context.parameters(), 3e-5, "Phase B candidate context")
    assert_lr(mapping, net.unified_density_decoder.candidate_input.parameters(), 3e-5, "Phase B candidate input")
    assert_lr(mapping, net.unified_density_decoder.head.parameters(), 3e-5, "Phase B joint density head")
    assert_lr(mapping, net.unified_density_decoder.candidate_norm.parameters(), 5e-6, "Phase B candidate norm")
    print("[PASS] Phase B warms the candidate-conditioned joint head")
    del module
    gc.collect()


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def module_grad_norm(module: torch.nn.Module) -> float:
    values = [parameter.grad.detach().float().norm() for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.stack(values).norm()) if values else 0.0


def forward_smoke(cfg: Any, phase: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = TrainingLightningModule(cfg).to(device)
    module.train()
    module.net.set_training_epoch(1)
    module.net.set_training_step(500)

    dm = TrainingDataModule(cfg)
    dm.setup(stage="fit")
    item = dm.train_dataset[0]
    batch = default_collate([item])
    batch.pop("gt_voxels", None)
    batch = move_to_device(batch, device)

    surface = batch["surface_measurements_packed"]
    query_mm = batch["query_coordinates_mm"]
    out = module.net(
        surface,
        query_mm,
        detector_valid_mask=batch.get("detector_valid_mask"),
        depth_maps=batch.get("depth_maps"),
        batch=batch,
        return_diagnostics=False,
    )
    pred = out["density"]
    aux = out.get("aux_outputs", {})
    losses = module.ssq_loss_func(
        pred,
        batch["point_densities"].unsqueeze(-1),
        aux,
        gt_voxels=None,
        points_ijk=batch.get("points_ijk"),
        query_component_ids=batch.get("query_component_ids"),
        gt_component_centers_mm=batch.get("gt_component_centers_mm"),
        gt_component_valid_mask=batch.get("gt_component_valid_mask"),
    )
    total = losses["total_loss"]
    check(torch.isfinite(total).item(), f"{phase} forward loss is nonfinite")
    total.backward()
    check(torch.isfinite(pred).all().item(), f"{phase} prediction contains NaN/Inf")

    finite_losses = {
        key: float(value.detach().cpu())
        for key, value in losses.items()
        if torch.is_tensor(value) and value.numel() == 1 and torch.isfinite(value).item()
    }
    print(f"[PASS] {phase} differentiable smoke on {device}: {finite_losses}")
    print(
        f"       gradients: encoder={module_grad_norm(module.net.surface_encoder):.3e}, "
        f"evidence={module_grad_norm(module.net.view_candidate_evidence):.3e}, "
        f"constructor={module_grad_norm(module.net.diverse_candidate_constructor):.3e}, "
        f"candidate_projection={module_grad_norm(module.net.complementary_aggregation.candidate_projection):.3e}, "
        f"candidate_context={module_grad_norm(module.net.unified_density_decoder.candidate_context):.3e}, "
        f"head={module_grad_norm(module.net.unified_density_decoder.head):.3e}"
    )
    del batch, item, dm, module, out, pred, aux, losses, total
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    for name, epochs in (
        ("phase_a", args.phase_a_epochs),
        ("phase_b", args.phase_b_epochs),
        ("full", args.full_epochs),
    ):
        check(epochs > 0, f"{name} must contain at least one epoch")
    check(args.train_queries > 0 and args.hypothesis_points > 0, "query counts must be positive")

    activate_phsa_sample_level_hypotheses()
    cfg_a = make_cfg(args, "phase_a")
    audit_splits(cfg_a, args)
    del cfg_a
    gc.collect()

    cfg_b = make_cfg(args, "phase_b")
    audit_phase_b_optimizer(cfg_b)
    del cfg_b
    gc.collect()

    cfg_c = make_cfg(args, "full")
    audit_aggregation_and_full_optimizer(cfg_c, args)
    del cfg_c
    gc.collect()

    if not args.skip_forward_smoke:
        forward_smoke(make_cfg(args, "phase_a", smoke=True), "phase_a")
        forward_smoke(make_cfg(args, "phase_b", smoke=True), "phase_b")

    print("\nPHSA PREFLIGHT: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
