"""
Training callbacks for lightweight progress logging.
"""

from pathlib import Path
import time
from typing import Optional

import pytorch_lightning as pl

from ivf.utils.logging import get_logger


class StepProgressLogger(pl.Callback):
    """
    Log a lightweight progress line every N steps to show training is active.
    """

    def __init__(self) -> None:
        self._last_time: Optional[float] = None
        self._last_step: Optional[int] = None

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        now = time.time()
        self._last_time = now
        self._last_step = trainer.global_step


class BestMetricCheckpoint(pl.Callback):
    """
    Save the best checkpoint based on a primary metric, with fallback to another metric.
    """

    def __init__(
        self,
        ckpt_path: Path,
        primary_metric: str,
        fallback_metric: str,
        primary_mode: str = "max",
        fallback_mode: str = "min",
    ) -> None:
        self.ckpt_path = Path(ckpt_path)
        self.primary_metric = primary_metric
        self.fallback_metric = fallback_metric
        self.primary_mode = primary_mode
        self.fallback_mode = fallback_mode
        self.best_score: Optional[float] = None
        self.best_metric_name: Optional[str] = None

    def _select_metric(self, metrics) -> tuple[Optional[str], Optional[float], Optional[str]]:
        if self.primary_metric in metrics:
            value = metrics[self.primary_metric]
            try:
                score = float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
                return self.primary_metric, score, self.primary_mode
            except (TypeError, ValueError):
                pass
        if self.fallback_metric in metrics:
            value = metrics[self.fallback_metric]
            try:
                score = float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
                return self.fallback_metric, score, self.fallback_mode
            except (TypeError, ValueError):
                pass
        return None, None, None

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        metric_name, score, mode = self._select_metric(trainer.callback_metrics)
        if metric_name is None or score is None or mode is None:
            return

        improved = False
        if self.best_score is None:
            improved = True
        elif mode == "max" and score > self.best_score:
            improved = True
        elif mode == "min" and score < self.best_score:
            improved = True

        if improved:
            trainer.save_checkpoint(str(self.ckpt_path))
            self.best_score = score
            self.best_metric_name = metric_name
            get_logger("ivf").info(
                "Saved best checkpoint to %s (metric=%s score=%.4f).",
                self.ckpt_path,
                metric_name,
                score,
            )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        if getattr(trainer, "global_rank", 0) != 0:
            return

        interval = max(1, int(getattr(trainer, "log_every_n_steps", 50)))
        is_last = (batch_idx + 1) >= getattr(trainer, "num_training_batches", 0)
        if not is_last and (batch_idx + 1) % interval != 0:
            return

        now = time.time()
        last_time = self._last_time or now
        last_step = self._last_step if self._last_step is not None else trainer.global_step
        steps_since = max(1, trainer.global_step - last_step)
        elapsed = max(1e-6, now - last_time)
        step_time = elapsed / steps_since

        num_batches = getattr(trainer, "num_training_batches", 0)
        remaining = max(0, num_batches - (batch_idx + 1))
        eta = remaining * step_time

        lr = None
        if trainer.optimizers:
            try:
                lr = trainer.optimizers[0].param_groups[0].get("lr")
            except (IndexError, AttributeError, KeyError, TypeError):
                lr = None

        metrics = trainer.callback_metrics
        loss = None
        for key in ("train/loss_step", "train/loss"):
            if key in metrics:
                value = metrics[key]
                try:
                    loss = float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
                except (TypeError, ValueError):
                    loss = None
                break

        message = (
            f"[epoch {trainer.current_epoch + 1}] step {trainer.global_step} "
            f"batch {batch_idx + 1}/{num_batches} "
            f"lr={lr:.6f} " if lr is not None else
            f"[epoch {trainer.current_epoch + 1}] step {trainer.global_step} "
            f"batch {batch_idx + 1}/{num_batches} "
        )
        if loss is not None:
            message += f"loss={loss:.4f} "
        message += f"step_time={step_time:.3f}s eta={eta:.1f}s"
        get_logger("ivf").info(message)

        self._last_time = now
        self._last_step = trainer.global_step
