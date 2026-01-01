"""
Lightning module for multi-phase IVF training.
"""

import time
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from ivf.metrics import build_morphology_metrics, build_quality_metrics, build_stage_metrics
from ivf.models.freezing import freeze_encoder, progressive_unfreeze
from ivf.utils.guardrails import assert_no_day_feature, assert_no_segmentation_inputs
from ivf.utils.logging import get_logger


class MultiTaskLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        phase: str,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_weights: Optional[Dict[str, float]] = None,
        freeze_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.phase = phase
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_weights = loss_weights or {"morph": 1.0, "stage": 1.0, "quality": 1.0}
        self.freeze_config = freeze_config or {}

        self._apply_phase_freeze(initial=True)

        self.morph_metrics = nn.ModuleDict(build_morphology_metrics())
        self.stage_metrics = nn.ModuleDict(build_stage_metrics())
        self.quality_metrics = nn.ModuleDict(build_quality_metrics())

        self._epoch_start_time = None
        self._next_progress_pct = None

        self.save_hyperparameters(ignore=["model"])

    def _apply_phase_freeze(self, initial: bool = False) -> None:
        logger = get_logger("ivf")
        def _set_trainable(module: nn.Module, trainable: bool) -> None:
            for p in module.parameters():
                p.requires_grad = trainable

        if self.phase == "morph":
            _set_trainable(self.model.encoder, True)
            _set_trainable(self.model.morph, True)
            _set_trainable(self.model.stage, False)
            _set_trainable(self.model.quality, False)
            logger.info("Phase morph: encoder+morph trainable; stage+quality frozen.")
        elif self.phase == "stage":
            _set_trainable(self.model.encoder, True)
            _set_trainable(self.model.morph, False)
            _set_trainable(self.model.stage, True)
            _set_trainable(self.model.quality, False)
            if initial:
                freeze_ratio = self.freeze_config.get("stage_start_ratio", 0.8)
                freeze_encoder(self.model, ratio=freeze_ratio)
                logger.info("Phase stage: initial freeze ratio=%s", freeze_ratio)
        elif self.phase == "joint":
            _set_trainable(self.model.encoder, True)
            _set_trainable(self.model.morph, True)
            _set_trainable(self.model.stage, True)
            _set_trainable(self.model.quality, False)
            logger.info("Phase joint: encoder+morph+stage trainable; quality frozen.")
        elif self.phase == "quality":
            freeze_encoder(self.model, ratio=1.0)
            _set_trainable(self.model.morph, False)
            _set_trainable(self.model.stage, False)
            _set_trainable(self.model.quality, True)
            logger.info("Phase quality: encoder+morph+stage frozen; quality trainable.")
        else:
            raise ValueError(f"Unsupported phase: {self.phase}")

    def on_train_epoch_start(self) -> None:
        if self.phase == "stage":
            schedule = self.freeze_config.get("stage_schedule")
            if schedule:
                ratio = progressive_unfreeze(self.model, epoch=self.current_epoch, schedule=schedule)
                self.log("train/freeze_ratio", ratio, on_epoch=True, prog_bar=False)
                get_logger("ivf").info("Stage phase epoch %s: freeze ratio=%s", self.current_epoch, ratio)
        self._epoch_start_time = time.time()
        self._next_progress_pct = 25.0

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if not self.trainer or getattr(self.trainer, "sanity_checking", False):
            return
        num_batches = getattr(self.trainer, "num_training_batches", None)
        if not isinstance(num_batches, int) or num_batches <= 0:
            return
        if self._next_progress_pct is None:
            return

        progress = (batch_idx + 1) / num_batches * 100.0
        is_last = (batch_idx + 1) >= num_batches
        if progress < self._next_progress_pct and not is_last:
            return

        elapsed = (time.time() - self._epoch_start_time) if self._epoch_start_time else 0.0
        it_per_s = (batch_idx + 1) / elapsed if elapsed > 0 else 0.0

        get_logger("ivf").info(
            "[epoch %s] progress=%.1f%% it/s=%.2f",
            self.current_epoch + 1,
            min(progress, 100.0),
            it_per_s,
        )

        while self._next_progress_pct is not None and progress >= self._next_progress_pct:
            self._next_progress_pct += 25.0

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

    def _guardrails(self, batch: Dict) -> None:
        assert_no_day_feature(batch)
        assert_no_segmentation_inputs(batch)

    def _masked_ce(self, logits: torch.Tensor, targets: torch.Tensor, weight: float) -> Optional[torch.Tensor]:
        mask = targets >= 0
        if not mask.any():
            return None
        return F.cross_entropy(logits[mask], targets[mask]) * weight

    def _compute_losses(self, outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        losses = {}

        if self.phase in {"morph", "joint"}:
            loss_exp = self._masked_ce(outputs["morph"]["exp"], targets["exp"], self.loss_weights.get("morph", 1.0))
            loss_icm = self._masked_ce(outputs["morph"]["icm"], targets["icm"], self.loss_weights.get("morph", 1.0))
            loss_te = self._masked_ce(outputs["morph"]["te"], targets["te"], self.loss_weights.get("morph", 1.0))
            morph_losses = [l for l in [loss_exp, loss_icm, loss_te] if l is not None]
            if morph_losses:
                losses["morphology"] = sum(morph_losses)

        if self.phase in {"stage", "joint"}:
            stage_loss = self._masked_ce(outputs["stage"], targets["stage"], self.loss_weights.get("stage", 1.0))
            if stage_loss is not None:
                losses["stage"] = stage_loss

        if self.phase == "quality":
            quality_loss = self._masked_ce(outputs["quality"], targets["quality"], self.loss_weights.get("quality", 1.0))
            if quality_loss is not None:
                losses["quality"] = quality_loss

        total = sum(losses.values()) if losses else torch.tensor(0.0, device=outputs["features"].device)
        losses["total"] = total
        return losses

    def training_step(self, batch: Dict, batch_idx: int):
        self._guardrails(batch)
        outputs = self.model(batch["image"])
        losses = self._compute_losses(outputs, batch["targets"])
        self.log("train/loss", losses["total"], on_step=True, on_epoch=True, prog_bar=True)
        if "morphology" in losses:
            self.log("train/morph_loss", losses["morphology"], on_step=True, on_epoch=True, prog_bar=False)
        if "stage" in losses:
            self.log("train/stage_loss", losses["stage"], on_step=True, on_epoch=True, prog_bar=False)
        if "quality" in losses:
            self.log("train/quality_loss", losses["quality"], on_step=True, on_epoch=True, prog_bar=False)
        return losses["total"]

    def validation_step(self, batch: Dict, batch_idx: int):
        self._guardrails(batch)
        outputs = self.model(batch["image"])
        losses = self._compute_losses(outputs, batch["targets"])
        self.log("val/loss", losses["total"], on_step=False, on_epoch=True, prog_bar=True)
        if "morphology" in losses:
            self.log("val/morph_loss", losses["morphology"], on_epoch=True, prog_bar=False)
        if "stage" in losses:
            self.log("val/stage_loss", losses["stage"], on_epoch=True, prog_bar=False)
        if "quality" in losses:
            self.log("val/quality_loss", losses["quality"], on_epoch=True, prog_bar=False)

        targets = batch["targets"]

        if self.phase in {"morph", "joint"}:
            for key, metric in self.morph_metrics.items():
                head = key.split("_")[0]
                t = targets[head]
                mask = t >= 0
                if mask.any():
                    preds = outputs["morph"][head].argmax(dim=-1)
                    metric.update(preds[mask], t[mask])
                    self.log(f"val/{key}", metric, on_epoch=True, prog_bar=False)

        if self.phase in {"stage", "joint"}:
            t = targets["stage"]
            mask = t >= 0
            if mask.any():
                preds = outputs["stage"].argmax(dim=-1)
                for key, metric in self.stage_metrics.items():
                    metric.update(preds[mask], t[mask])
                    self.log(f"val/{key}", metric, on_epoch=True, prog_bar=False)

        if self.phase == "quality":
            t = targets["quality"]
            mask = t >= 0
            if mask.any():
                probs = F.softmax(outputs["quality"], dim=-1)[:, 1]
                for key, metric in self.quality_metrics.items():
                    metric.update(probs[mask], t[mask])
                    self.log(f"val/{key}", metric, on_epoch=True, prog_bar=False)

    def on_validation_epoch_end(self) -> None:
        if self.trainer and getattr(self.trainer, "sanity_checking", False):
            return

        metrics = self.trainer.callback_metrics if self.trainer else {}

        def _value(key):
            if key not in metrics:
                return None
            val = metrics[key]
            if isinstance(val, torch.Tensor):
                return float(val.detach().cpu())
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        def _append(parts, label, key):
            value = _value(key)
            if value is not None:
                parts.append(f"{label}={value:.4f}")

        parts = [f"[epoch {self.current_epoch + 1}]"]
        train_loss = _value("train/loss_epoch")
        if train_loss is None:
            train_loss = _value("train/loss")
        if train_loss is not None:
            parts.append(f"train_loss={train_loss:.4f}")

        _append(parts, "val_loss", "val/loss")

        if self.phase in {"morph", "joint"}:
            _append(parts, "val_exp_acc", "val/exp_acc")
            _append(parts, "val_icm_acc", "val/icm_acc")
            _append(parts, "val_te_acc", "val/te_acc")
        if self.phase in {"stage", "joint"}:
            _append(parts, "val_stage_acc", "val/stage_acc")
            _append(parts, "val_stage_f1", "val/stage_f1")
        if self.phase == "quality":
            _append(parts, "val_auroc", "val/quality_auroc")
            _append(parts, "val_auprc", "val/quality_auprc")
            _append(parts, "val_f1", "val/quality_f1")
            _append(parts, "val_acc", "val/quality_acc")

        if len(parts) > 1:
            get_logger("ivf").info(" ".join(parts))

        for metric in list(self.morph_metrics.values()) + list(self.stage_metrics.values()) + list(self.quality_metrics.values()):
            metric.reset()
