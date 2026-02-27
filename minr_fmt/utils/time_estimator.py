from __future__ import annotations

import time

import pytorch_lightning as pl


class TrainingTimeEstimator(pl.Callback):
    """Log an estimated remaining training time to the progress bar."""

    def __init__(self, log_every_n_steps: int = 10, ema_alpha: float = 0.05):
        self.log_every_n_steps = int(log_every_n_steps)
        self.ema_alpha = float(ema_alpha)
        self._fit_start_t: float | None = None
        self._last_step_t: float | None = None
        self._last_global_step: int = -1
        self._ema_step_s: float | None = None

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        now = time.perf_counter()
        self._fit_start_t = now
        self._last_step_t = now
        self._last_global_step = int(trainer.global_step)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        # Update only when global_step advances (respects grad accumulation).
        gs = int(trainer.global_step)
        if gs == self._last_global_step:
            return

        now = time.perf_counter()
        if self._last_step_t is not None:
            dt = now - self._last_step_t
            if self._ema_step_s is None:
                self._ema_step_s = dt
            else:
                a = self.ema_alpha
                self._ema_step_s = (1 - a) * self._ema_step_s + a * dt

        self._last_step_t = now
        self._last_global_step = gs

        if self._fit_start_t is None or self._ema_step_s is None:
            return

        if self.log_every_n_steps > 1 and (gs % self.log_every_n_steps) != 0:
            return

        total_steps = int(getattr(trainer, "estimated_stepping_batches", 0) or 0)
        if total_steps <= 0:
            return

        remaining_steps = max(total_steps - gs, 0)
        eta_s = remaining_steps * self._ema_step_s
        elapsed_s = now - self._fit_start_t

        # Shown in progress bar via RichProgressBar (logger disabled).
        pl_module.log("eta_h", eta_s / 3600.0, prog_bar=True, on_step=True, logger=False)
        pl_module.log("elapsed_h", elapsed_s / 3600.0, prog_bar=True, on_step=True, logger=False)
