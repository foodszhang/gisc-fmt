#!/usr/bin/env python3
"""Full-volume evaluator for reproduced Phase-A and factorized checkpoints."""

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
    attach_factorized_unified_output,
    load_factorized_or_historical_checkpoint,
)
from minr_fmt.phsa_sample_level import activate_phsa_sample_level_hypotheses


activate_phsa_sample_level_hypotheses()


def load_factorized_net(cfg, ckpt_path: Path | None, device: torch.device):
    view_cfg = cfg.model.ssq_fmt.view_complementary
    factor_cfg = view_cfg.get("factorized_reconstruction", {}) or {}
    if str(view_cfg.get("training_phase", "")) != "phase_a":
        raise ValueError("morphology-amplitude evaluation requires training_phase=phase_a")
    if str(view_cfg.get("ablation", "")) != "full":
        raise ValueError("morphology-amplitude evaluation requires ablation=full")

    net = ModelFactory.create_model(cfg.model.name, config=cfg)
    attach_factorized_unified_output(net, factor_cfg)
    net.to(device)
    if ckpt_path is None or str(ckpt_path).lower() in {"", "none", "null"}:
        net.eval()
        return net
    report = load_factorized_or_historical_checkpoint(net, ckpt_path)
    print(f"[morphology-amplitude-eval] checkpoint load: {report}")
    net.eval()
    return net


base_eval.load_net = load_factorized_net

if __name__ == "__main__":
    base_eval.main()
