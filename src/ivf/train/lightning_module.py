"""
Lightning module for multi-phase IVF training.
"""

import time
import sys
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset
from torchmetrics.classification import MulticlassAccuracy

from ivf.data.datasets import BaseImageDataset, IGNORE_INDEX
from ivf.data.label_schema import ICM_CLASSES, TE_CLASSES
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
        morph_loss_reduction: str = "mean",
        quality_pos_weight: Optional[float] = None,
        live_epoch_line: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.phase = phase
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_weights = loss_weights or {"morph": 1.0, "stage": 1.0, "quality": 1.0}
        self.freeze_config = freeze_config or {}
        self.morph_loss_reduction = morph_loss_reduction
        self.quality_pos_weight = quality_pos_weight
        self.live_epoch_line = live_epoch_line
        self._epoch_start_time = None
        self._val_pred_counts = None
        self._val_true_counts = None
        self.icm_class_weight = None
        self.te_class_weight = None
        self.icm_num_classes = len(ICM_CLASSES)
        self.te_num_classes = len(TE_CLASSES)

        if self.morph_loss_reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported morph_loss_reduction: {self.morph_loss_reduction}")

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
        if self.live_epoch_line:
            self._epoch_start_time = time.time()

    def on_fit_start(self) -> None:
        if self.phase not in {"morph", "joint"}:
            return
        if self.icm_class_weight is not None or self.te_class_weight is not None:
            return
        self._setup_morph_class_weights()

    def on_validation_epoch_start(self) -> None:
        if self.phase in {"morph", "joint"}:
            self._val_pred_counts = {
                "icm": torch.zeros(self.icm_num_classes, dtype=torch.long),
                "te": torch.zeros(self.te_num_classes, dtype=torch.long),
            }
            self._val_true_counts = {
                "icm": torch.zeros(self.icm_num_classes, dtype=torch.long),
                "te": torch.zeros(self.te_num_classes, dtype=torch.long),
            }

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if not self.live_epoch_line:
            return
        if not self.trainer or getattr(self.trainer, "sanity_checking", False):
            return
        if getattr(self.trainer, "global_rank", 0) != 0:
            return
        num_batches = getattr(self.trainer, "num_training_batches", None)
        if not isinstance(num_batches, int) or num_batches <= 0:
            return

        elapsed = (time.time() - self._epoch_start_time) if self._epoch_start_time else 0.0
        it_per_s = (batch_idx + 1) / elapsed if elapsed > 0 else 0.0
        progress = (batch_idx + 1) / num_batches * 100.0

        sys.stdout.write(
            f"\r[epoch {self.current_epoch + 1}] progress={progress:5.1f}% it/s={it_per_s:6.2f}"
        )
        sys.stdout.flush()
        if (batch_idx + 1) >= num_batches:
            sys.stdout.write("\n")
            sys.stdout.flush()
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

    def _masked_ce(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor],
        weight: float,
        class_weight: Optional[torch.Tensor] = None,
        num_classes: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if mask is None:
            mask = targets >= 0
        else:
            mask = mask > 0
        if num_classes is not None:
            mask = mask & (targets < num_classes)
            logits = logits[:, :num_classes]
        if not mask.any():
            return None
        if class_weight is not None:
            class_weight = class_weight.to(logits.device)
        return F.cross_entropy(logits[mask], targets[mask], weight=class_weight) * weight

    def _compute_losses(self, outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        losses = {}

        if self.phase in {"morph", "joint"}:
            loss_exp = self._masked_ce(
                outputs["morph"]["exp"],
                targets["exp"],
                targets.get("exp_mask"),
                self.loss_weights.get("morph", 1.0),
            )
            loss_icm = self._masked_ce(
                outputs["morph"]["icm"],
                targets["icm"],
                targets.get("icm_mask"),
                self.loss_weights.get("morph", 1.0),
                class_weight=self.icm_class_weight,
                num_classes=self.icm_num_classes,
            )
            loss_te = self._masked_ce(
                outputs["morph"]["te"],
                targets["te"],
                targets.get("te_mask"),
                self.loss_weights.get("morph", 1.0),
                class_weight=self.te_class_weight,
                num_classes=self.te_num_classes,
            )
            morph_losses = [l for l in [loss_exp, loss_icm, loss_te] if l is not None]
            if morph_losses:
                total = sum(morph_losses)
                if self.morph_loss_reduction == "mean":
                    total = total / len(morph_losses)
                losses["morphology"] = total

        if self.phase in {"stage", "joint"}:
            stage_loss = self._masked_ce(
                outputs["stage"],
                targets["stage"],
                None,
                self.loss_weights.get("stage", 1.0),
            )
            if stage_loss is not None:
                losses["stage"] = stage_loss

        if self.phase == "quality":
            weight = self.loss_weights.get("quality", 1.0)
            if self.quality_pos_weight is not None:
                logits = outputs["quality"][:, 1] - outputs["quality"][:, 0]
                t = targets["quality"].float()
                mask = t >= 0
                if mask.any():
                    pos_weight = torch.tensor(self.quality_pos_weight, device=logits.device)
                    loss = F.binary_cross_entropy_with_logits(logits[mask], t[mask], pos_weight=pos_weight)
                    losses["quality"] = loss * weight
            else:
                quality_loss = self._masked_ce(
                    outputs["quality"],
                    targets["quality"],
                    None,
                    weight,
                )
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
                mask = targets.get(f"{head}_mask")
                mask = mask > 0 if mask is not None else t >= 0
                num_classes = self.icm_num_classes if head == "icm" else self.te_num_classes if head == "te" else None
                if num_classes is not None:
                    mask = mask & (t < num_classes)
                if mask.any():
                    logits = outputs["morph"][head]
                    if num_classes is not None:
                        logits = logits[:, :num_classes]
                    preds = logits.argmax(dim=-1)
                    metric.update(preds[mask], t[mask])
                    self.log(f"val/{key}", metric, on_epoch=True, prog_bar=False)

            if self._val_pred_counts is not None:
                for head, classes in (("icm", ICM_CLASSES), ("te", TE_CLASSES)):
                    t = targets[head]
                    mask = targets.get(f"{head}_mask")
                    num_classes = self.icm_num_classes if head == "icm" else self.te_num_classes
                    mask = mask > 0 if mask is not None else t >= 0
                    mask = mask & (t < num_classes)
                    if mask.any():
                        mask_cpu = mask.detach().cpu()
                        logits = outputs["morph"][head].detach().cpu()[:, :num_classes]
                        preds = logits.argmax(dim=-1)
                        counts = torch.bincount(preds[mask_cpu], minlength=num_classes)
                        self._val_pred_counts[head] += counts
                        true_counts = torch.bincount(t.detach().cpu()[mask_cpu], minlength=num_classes)
                        self._val_true_counts[head] += true_counts

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
                logits = outputs["quality"][:, 1] - outputs["quality"][:, 0]
                probs = torch.sigmoid(logits)
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

        if self._val_pred_counts and self.phase in {"morph", "joint"}:
            logger = get_logger("ivf")
            for head, classes in (("icm", ICM_CLASSES[: self.icm_num_classes]), ("te", TE_CLASSES[: self.te_num_classes])):
                counts = self._val_pred_counts.get(head)
                true_counts = self._val_true_counts.get(head) if self._val_true_counts else None
                if counts is None:
                    continue
                count_dict = {cls: int(counts[idx]) for idx, cls in enumerate(classes)}
                logger.info("Validation %s prediction counts: %s", head, count_dict)
                if true_counts is not None:
                    true_dict = {cls: int(true_counts[idx]) for idx, cls in enumerate(classes)}
                    logger.info("Validation %s true counts: %s", head, true_dict)
                nonzero = int((counts > 0).sum())
                if nonzero <= 1:
                    logger.warning("Prediction collapse detected for %s head.", head)

        for metric in list(self.morph_metrics.values()) + list(self.stage_metrics.values()) + list(self.quality_metrics.values()):
            metric.reset()

    def _setup_morph_class_weights(self) -> None:
        if not self.trainer or not hasattr(self.trainer, "datamodule"):
            return
        dataset = getattr(self.trainer.datamodule, "train_dataset", None)
        if dataset is None:
            return

        counts = {
            "icm": torch.zeros(len(ICM_CLASSES), dtype=torch.long),
            "te": torch.zeros(len(TE_CLASSES), dtype=torch.long),
        }

        def _iter_samples(ds):
            if isinstance(ds, BaseImageDataset):
                for sample in ds.samples:
                    yield sample
            elif isinstance(ds, ConcatDataset):
                for subset in ds.datasets:
                    yield from _iter_samples(subset)

        for sample in _iter_samples(dataset):
            targets = sample.get("targets", {})
            for head in ("icm", "te"):
                label = targets.get(head, IGNORE_INDEX)
                mask = targets.get(f"{head}_mask", 0)
                if mask and label is not None and label >= 0:
                    if label < counts[head].numel():
                        counts[head][int(label)] += 1

        logger = get_logger("ivf")
        logger.info("Morph train icm counts: %s", {cls: int(counts["icm"][i]) for i, cls in enumerate(ICM_CLASSES)})
        logger.info("Morph train te counts: %s", {cls: int(counts["te"][i]) for i, cls in enumerate(TE_CLASSES)})

        self.icm_num_classes = 2 if counts["icm"][2] == 0 else 3
        self.te_num_classes = 2 if counts["te"][2] == 0 else 3
        if self.icm_num_classes == 2:
            logger.warning("ICM class C missing; using 2-class (A/B) loss and metrics.")
        if self.te_num_classes == 2:
            logger.warning("TE class C missing; using 2-class (A/B) loss and metrics.")
        logger.info(
            "Morph heads for loss/metrics: icm_num_classes=%s te_num_classes=%s.",
            self.icm_num_classes,
            self.te_num_classes,
        )

        def _compute_weights(head: str, num_classes: int) -> torch.Tensor:
            head_counts = counts[head][:num_classes].float()
            total = head_counts.sum()
            weights = torch.ones_like(head_counts)
            for i, c in enumerate(head_counts):
                if c > 0:
                    weights[i] = total / (num_classes * c)
                else:
                    weights[i] = 0.0
            return weights

        self.icm_class_weight = _compute_weights("icm", self.icm_num_classes)
        self.te_class_weight = _compute_weights("te", self.te_num_classes)

        self.morph_metrics["icm_acc"] = MulticlassAccuracy(num_classes=self.icm_num_classes)
        self.morph_metrics["te_acc"] = MulticlassAccuracy(num_classes=self.te_num_classes)
