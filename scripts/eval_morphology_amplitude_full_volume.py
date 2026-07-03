#!/usr/bin/env python3
"""Full-volume evaluator that restores the optional factorized decoder head."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.eval_full_volume_fmt_simgen as base_eval
from minr_fmt.model_factory import ModelFactory


def load_factorized_net(cfg, ckpt_path: Path | None, device: torch.device):
    net = ModelFactory.create_model(cfg.model.name, config=cfg).to(device)
    factor_cfg = (
        cfg.model.ssq_fmt.view_complementary.get("factorized_reconstruction", {}) or {}
    )
    if bool(factor_cfg.get("enabled", False)):
        decoder = getattr(net, "unified_density_decoder", None)
        if decoder is None or not hasattr(decoder, "enable_factorized_output"):
            raise RuntimeError("Configured factorized reconstruction has no compatible decoder")
        decoder.enable_factorized_output(
            compose_density=bool(factor_cfg.get("compose_density", True)),
            support_init_logit=float(factor_cfg.get("support_init_logit", 8.0)),
        )
        net.to(device)
    if ckpt_path is None or str(ckpt_path).lower() in {"", "none", "null"}:
        net.eval()
        return net
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    net_state = {
        key.removeprefix("net."): value
        for key, value in state.items()
        if key.startswith("net.")
    }
    if not net_state:
        net_state = state
    missing, unexpected = net.load_state_dict(net_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Strict factorized checkpoint load failed: missing={missing[:10]} "
            f"unexpected={unexpected[:10]}"
        )
    net.eval()
    return net


base_eval.load_net = load_factorized_net

if __name__ == "__main__":
    base_eval.main()
