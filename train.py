"""Training entrypoint for SSQ-FMT/GISC-FMT (Hydra + PyTorch Lightning).

Common usage (with uv):
  uv run python train.py fit
  uv run python train.py fit exp=debug
  uv run python train.py validate ckpt_path=/abs/path/to.ckpt
  uv run python train.py test ckpt_path=/abs/path/to.ckpt

Notes:
- All hyperparameters are managed by Hydra under ./configs.
- The LightningModule/DataModule implementations live in minr_fmt/.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from minr_fmt.datamodule import TrainingDataModule
from minr_fmt.module import TrainingLightningModule
from minr_fmt.utils.hydra_utils import get_git_info
from minr_fmt.utils.logging_utils import rank_zero_log, setup_logger
from minr_fmt.utils.seed_utils import set_seed

logger = logging.getLogger(__name__)


def _ensure_output_dirs(cfg: DictConfig) -> None:
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.paths.config_dir, exist_ok=True)


def _save_metadata(cfg: DictConfig) -> None:
    # Save resolved config
    config_path = Path(cfg.paths.config_dir) / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)
    rank_zero_log(logger, "info", f"Resolved config saved to {config_path}")

    # Save git info (best-effort)
    if "exp" in cfg and cfg.exp.get("log_git_info", True):
        try:
            info = get_git_info()
            git_file = Path(cfg.paths.output_dir) / "git_info.txt"
            git_file.write_text(
                f"Commit: {info['commit']}\nBranch: {info['branch']}\nDirty: {info['dirty']}\n"
            )
            rank_zero_log(logger, "info", f"Git: {info['commit'][:7]} on {info['branch']}")
        except Exception as e:
            rank_zero_log(logger, "warning", f"Could not get git info: {e}")


def _instantiate_callbacks(cfg: DictConfig) -> list:
    callbacks = []
    if "callbacks" in cfg and cfg.callbacks is not None:
        for _, cb_cfg in cfg.callbacks.items():
            if cb_cfg is not None and "_target_" in cb_cfg:
                callbacks.append(hydra.utils.instantiate(cb_cfg))
    return callbacks


def _instantiate_logger(cfg: DictConfig):
    if "logger" not in cfg or cfg.logger is None:
        return None

    target = cfg.logger.get("_target_") if isinstance(cfg.logger, DictConfig) else None
    if target in (None, "", "null"):
        return None

    # Allow logger name to default to output folder name
    logger_cfg = cfg.logger.copy()
    if "name" in logger_cfg and logger_cfg.name is None:
        logger_cfg.name = Path(cfg.paths.output_dir).name

    return hydra.utils.instantiate(logger_cfg)


def run(cfg: DictConfig) -> Optional[float]:
    """Run one of: fit / validate / test (selected by cfg.task)."""
    _ensure_output_dirs(cfg)

    set_seed(cfg.seed, deterministic=cfg.deterministic, benchmark=cfg.benchmark)

    # Enable Tensor Core friendly matmul for better performance on recent NVIDIA GPUs.
    # Safe no-op on CPU; can be overridden via env var.
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision(os.environ.get("TORCH_MATMUL_PRECISION", "medium"))

    if cfg.get("verbose", False):
        rank_zero_log(logger, "info", "Resolved config:\n" + OmegaConf.to_yaml(cfg))

    if cfg.get("exp") is None or cfg.exp.get("log_config", True):
        _save_metadata(cfg)

    dm = TrainingDataModule(cfg)
    model = TrainingLightningModule(cfg)

    callbacks = _instantiate_callbacks(cfg)
    logger_instance = _instantiate_logger(cfg)

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    # Remove non-Lightning kwargs (used by our code, not pl.Trainer)
    if isinstance(trainer_kwargs, dict):
        trainer_kwargs.pop("torch_compile", None)
        trainer_kwargs.pop("torch_compile_mode", None)

    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=callbacks,
        logger=logger_instance,
        default_root_dir=cfg.paths.output_dir,
    )

    ckpt_path = cfg.get("ckpt_path")
    task = str(cfg.get("task", "fit")).lower()
    weights_only = bool(cfg.get("ckpt_weights_only", True))

    if task == "fit" and weights_only and ckpt_path not in (None, "", "null"):
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        rank_zero_log(
            logger,
            "info",
            "Loaded model weights only from "
            f"{ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})",
        )
        ckpt_path = None

    # If user didn't specify ckpt_path for validate/test, try to use the last checkpoint.
    if task in {"validate", "test"} and (ckpt_path is None or str(ckpt_path) in {"", "null"}):
        last_ckpt = Path(str(cfg.paths.checkpoint_dir)) / "last.ckpt"
        if last_ckpt.exists():
            ckpt_path = str(last_ckpt)
            rank_zero_log(logger, "info", f"ckpt_path not set; using {ckpt_path}")
        else:
            rank_zero_log(
                logger,
                "warning",
                "ckpt_path not set and last.ckpt not found; running with random weights "
                "(metrics may be near zero)",
            )

    if task == "fit":
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path, weights_only=weights_only)
        metric = trainer.callback_metrics.get("val_dice")
        return metric.item() if hasattr(metric, "item") else None

    if task == "validate":
        trainer.validate(model, datamodule=dm, ckpt_path=ckpt_path, weights_only=weights_only)
        return None

    if task == "test":
        trainer.test(model, datamodule=dm, ckpt_path=ckpt_path, weights_only=weights_only)
        return None

    raise ValueError(f"Unknown task: {task} (expected: fit/validate/test)")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def hydra_main(cfg: DictConfig) -> Optional[float]:
    warnings.filterwarnings("ignore", category=UserWarning)
    setup_logger("ssq_fmt", level=logging.INFO)
    return run(cfg)


def _rewrite_positional_task(argv: list[str]) -> None:
    # Support: `uv run python train.py fit ...` (Hydra itself requires key=value overrides).
    if len(argv) >= 2 and argv[1] in {"fit", "validate", "test"}:
        task = argv.pop(1)
        argv.insert(1, f"task={task}")


if __name__ == "__main__":
    _rewrite_positional_task(sys.argv)
    hydra_main()
