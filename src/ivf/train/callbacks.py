"""
Training callbacks for lightweight progress logging.
"""

from pathlib import Path
import sys
import time
from typing import Optional

import pytorch_lightning as pl

from ivf.utils.logging import get_logger


class StepProgressLogger(pl.Callback):
    """
    Print a single-line progress update during training (Keras-like).
    """

    def __init__(
        self,
        single_line: bool = True,
        update_every_n_steps: Optional[int] = None,
        show_epoch_header: bool = True,
        bar_width: int = 30,
    ) -> None:
        self.single_line = single_line
        self.update_every_n_steps = update_every_n_steps
        self.show_epoch_header = show_epoch_header
        self.bar_width = bar_width
        self._epoch_start_time: Optional[float] = None
        self._last_len: int = 0

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._epoch_start_time = time.time()
        self._last_len = 0
        if self.show_epoch_header:
            sys.stdout.write(f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}\n")
            sys.stdout.flush()

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

        interval = self.update_every_n_steps or max(1, int(getattr(trainer, "log_every_n_steps", 50)))
        num_batches = getattr(trainer, "num_training_batches", 0)
        is_last = (batch_idx + 1) >= num_batches
        if not is_last and (batch_idx + 1) % interval != 0:
            return

        elapsed = max(1e-6, time.time() - (self._epoch_start_time or time.time()))
        it_per_s = (batch_idx + 1) / elapsed if elapsed > 0 else 0.0
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

        filled = int(self.bar_width * (batch_idx + 1) / max(1, num_batches))
        bar = "=" * max(0, filled - 1)
        bar += ">" if filled > 0 else ""
        bar += "." * max(0, self.bar_width - filled)
        ms_per_step = (1000.0 / it_per_s) if it_per_s > 0 else 0.0

        message = (
            f"{batch_idx + 1}/{num_batches} [{bar}] "
            f"- {elapsed:.0f}s {ms_per_step:.0f}ms/step"
        )
        if loss is not None:
            message += f" - loss: {loss:.4f}"

        if self.single_line:
            padding = " " * max(0, self._last_len - len(message))
            sys.stdout.write("\r" + message + padding)
            sys.stdout.flush()
            self._last_len = len(message)
            if is_last:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self._last_len = 0
        else:
            sys.stdout.write(message + "\n")
            sys.stdout.flush()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.single_line and self._last_len:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._last_len = 0


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
