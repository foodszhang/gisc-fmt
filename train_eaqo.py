from __future__ import annotations

import sys

import train as base_train
from minr_fmt.eaqo_module import EAQOTrainingLightningModule


def _cls(cfg):
    model_cfg = cfg.get("model", {})
    ssq_cfg = model_cfg.get("ssq_fmt", {})
    eaqo_cfg = ssq_cfg.get("eaqo", {})
    if str(model_cfg.get("name", "")).lower() == "ssq_fmt" and bool(eaqo_cfg.get("enabled", False)):
        return EAQOTrainingLightningModule
    return base_train._lightning_module_cls(cfg)


if __name__ == "__main__":
    base_train._lightning_module_cls = _cls
    base_train._rewrite_positional_task(sys.argv)
    base_train.hydra_main()
