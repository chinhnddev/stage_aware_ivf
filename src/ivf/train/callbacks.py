"""
Training callbacks for lightweight progress logging.
"""

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
