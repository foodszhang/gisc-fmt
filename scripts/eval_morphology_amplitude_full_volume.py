#!/usr/bin/env python3
"""Full-volume evaluator for the checkpoint-compatible Phase-A factorization."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.eval_full_volume_fmt_simgen as base_eval
from minr_fmt.model_factory import ModelFactory
from minr_fmt.network.morphology_amplitude import (
    activate_legacy_phase_a_decoder,
    load_legacy_phase_a_checkpoint,
)


def load_factorized_net(cfg, ckpt_path: Path | None, device: torch.device):
    net = ModelFactory.create_model(cfg.model.name, config=cfg)
    factor_cfg = (
        cfg.model.ssq_fmt.view_complementary.get("factorized_reconstruction", {}) or {}
    )
    if str(factor_cfg.get("base_decoder", "legacy_shared_logit")) != "legacy_shared_logit":
        raise ValueError("evaluation requires base_decoder=legacy_shared_logit")
    activate_legacy_phase_a_decoder(net, factor_cfg)
    net.to(device)
    if ckpt_path is None or str(ckpt_path).lower() in {"", "none", "null"}:
        net.eval()
        return net
    report = load_legacy_phase_a_checkpoint(
        net,
        ckpt_path,
        allowed_missing_prefixes=("unified_density_decoder.",),
    )
    print(f"[morphology-amplitude-eval] checkpoint load: {report}")
    net.eval()
    return net


base_eval.load_net = load_factorized_net

if __name__ == "__main__":
    base_eval.main()
