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
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from ivf.data.datasets import BaseImageDataset, IGNORE_INDEX
from ivf.data.label_schema import EXPANSION_CLASSES, ICM_CLASSES, TE_CLASSES
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
        use_class_weights: bool = False,
        class_weight_mode: str = "inverse_freq",
        q_loss: str = "smoothl1",
        q_aux_alpha: float = 0.0,
        q_freeze_backbone: bool = True,
        q_ranking_loss: bool = False,
        q_ranking_weight: float = 0.1,
        q_ranking_margin: float = 0.05,
        q_ranking_pairs: int = 256,
        quality_warmup_epochs: int = 0,
        quality_unfreeze_ratio: float = 0.0,
        quality_unfreeze_last_n_blocks: int = 0,
        quality_head_lr: Optional[float] = None,
        quality_encoder_lr_scale: float = 0.1,
        live_epoch_line: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.phase = phase
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_weights = loss_weights or {"morph": 1.0, "stage": 1.0, "quality": 1.0}
        if self.phase == "morph":
            self.loss_weights = dict(self.loss_weights)
            self.loss_weights["stage"] = 0.0
            self.loss_weights["quality"] = 0.0
            self.loss_weights["q"] = 0.0
            get_logger("ivf").info("Morph phase: forcing stage/quality loss weights to 0.")
        self.freeze_config = freeze_config or {}
        self.morph_loss_reduction = morph_loss_reduction
        self.quality_pos_weight = quality_pos_weight
        self.use_class_weights = use_class_weights
        self.class_weight_mode = class_weight_mode
        self.q_loss = q_loss
        self.q_aux_alpha = q_aux_alpha
        self.q_freeze_backbone = q_freeze_backbone
        self.q_ranking_loss = bool(q_ranking_loss)
        self.q_ranking_weight = float(q_ranking_weight)
        self.q_ranking_margin = float(q_ranking_margin)
        self.q_ranking_pairs = int(q_ranking_pairs)
        self.quality_warmup_epochs = int(quality_warmup_epochs)
        self.quality_unfreeze_ratio = float(quality_unfreeze_ratio)
        self.quality_unfreeze_last_n_blocks = int(quality_unfreeze_last_n_blocks)
        self.quality_head_lr = quality_head_lr
        self.quality_encoder_lr_scale = float(quality_encoder_lr_scale)
        self.live_epoch_line = live_epoch_line
        self._epoch_start_time = None
        self._val_pred_counts = None
        self._val_true_counts = None
        self._val_manual_correct = None
        self._val_manual_total = None
        self._val_counts = None
        self._collapse_streak = {"icm": 0, "te": 0}
        self._quality_confusion = None
        self._q_val_stats = None
        self.exp_class_weight = None
        self.icm_class_weight = None
        self.te_class_weight = None
        self.use_stage = bool(getattr(model, "use_stage", True))
        self.use_q_score = bool(getattr(model, "use_q_score", True))
        self.icm_num_classes = len(ICM_CLASSES)
        self.te_num_classes = len(TE_CLASSES)
        self.exp_class_counts = None
        self.icm_class_counts = None
        self.te_class_counts = None
        self.focal_gamma = 2.0
        self.use_focal_icm = False
        self.use_focal_te = False
        if not self.use_stage:
            get_logger("ivf").info("Stage disabled (use_stage=false). Stage head and losses are skipped.")

        if self.morph_loss_reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported morph_loss_reduction: {self.morph_loss_reduction}")
        if self.class_weight_mode not in {"inverse_freq"}:
            raise ValueError(f"Unsupported class_weight_mode: {self.class_weight_mode}")
        if self.q_loss not in {"smoothl1", "mse"}:
            raise ValueError(f"Unsupported q_loss: {self.q_loss}")
        if self.q_ranking_pairs < 0:
            raise ValueError("q_ranking_pairs must be >= 0.")
        if self.phase == "q" and not self.use_q_score:
            raise ValueError("Phase q requested but use_q_score=false. Enable use_q_score to train q_head.")

        self._apply_phase_freeze(initial=True)

        self.morph_metrics = nn.ModuleDict(build_morphology_metrics())
        self.stage_metrics = nn.ModuleDict(build_stage_metrics()) if self.use_stage else nn.ModuleDict()
        self.quality_metrics = nn.ModuleDict(build_quality_metrics())

        self._epoch_start_time = None
        self._next_progress_pct = None

        self.save_hyperparameters(ignore=["model"])

    def _set_trainable(self, module: nn.Module, trainable: bool) -> None:
        for p in module.parameters():
            p.requires_grad = trainable

    def _unfreeze_last_encoder_blocks(self, n_blocks: int) -> None:
        logger = get_logger("ivf")
        if n_blocks <= 0:
            return
        encoder = getattr(self.model, "encoder", None)
        blocks = getattr(encoder, "blocks", None)
        if blocks is None:
            logger.warning("Encoder blocks not found; unfreeze_last_n_blocks ignored.")
            return
        for block in list(blocks)[-n_blocks:]:
            self._set_trainable(block, True)
        if hasattr(encoder, "proj"):
            self._set_trainable(encoder.proj, True)

    def _apply_phase_freeze(self, initial: bool = False) -> None:
        logger = get_logger("ivf")

        if self.phase == "morph":
            self._set_trainable(self.model.encoder, True)
            self._set_trainable(self.model.morph, True)
            if self.use_stage and self.model.stage is not None:
                self._set_trainable(self.model.stage, False)
            self._set_trainable(self.model.quality, False)
            logger.info("EXP-1 Morphology: encoder+morph trainable; stage+quality frozen.")
        elif self.phase == "stage":
            self._set_trainable(self.model.encoder, True)
            self._set_trainable(self.model.morph, False)
            if self.use_stage and self.model.stage is not None:
                self._set_trainable(self.model.stage, True)
            self._set_trainable(self.model.quality, False)
            if initial:
                freeze_ratio = self.freeze_config.get("stage_start_ratio", 0.8)
                freeze_encoder(self.model, ratio=freeze_ratio)
                logger.info("EXP-2 Stage-aware: initial freeze ratio=%s", freeze_ratio)
        elif self.phase == "joint":
            self._set_trainable(self.model.encoder, True)
            self._set_trainable(self.model.morph, True)
            if self.use_stage and self.model.stage is not None:
                self._set_trainable(self.model.stage, True)
            self._set_trainable(self.model.quality, False)
            logger.info("EXP-3 Joint stabilization: encoder+morph+stage trainable; quality frozen.")
        elif self.phase == "quality":
            freeze_encoder(self.model, ratio=1.0)
            self._set_trainable(self.model.morph, False)
            if self.use_stage and self.model.stage is not None:
                self._set_trainable(self.model.stage, False)
            self._set_trainable(self.model.quality, True)
            logger.info("EXP-4 Quality: encoder+morph+stage frozen; quality trainable.")
        elif self.phase == "q":
            if self.q_freeze_backbone:
                freeze_encoder(self.model, ratio=1.0)
            else:
                self._set_trainable(self.model.encoder, True)
            self._set_trainable(self.model.morph, False)
            if self.use_stage and self.model.stage is not None:
                self._set_trainable(self.model.stage, False)
            self._set_trainable(self.model.quality, False)
            if self.model.q_head is not None:
                self._set_trainable(self.model.q_head, True)
            unfreeze_blocks = int(self.freeze_config.get("q_unfreeze_last_n_blocks", 0))
            if self.q_freeze_backbone and unfreeze_blocks > 0:
                self._unfreeze_last_encoder_blocks(unfreeze_blocks)
                logger.info("EXP-4Q: encoder frozen except last %s blocks; q_head trainable.", unfreeze_blocks)
            elif self.q_freeze_backbone:
                logger.info("EXP-4Q: encoder+morph+stage+quality frozen; q_head trainable.")
            else:
                logger.info("EXP-4Q: encoder unfrozen; q_head trainable.")
        else:
            raise ValueError(f"Unsupported phase: {self.phase}")

    def on_train_epoch_start(self) -> None:
        if self.phase == "stage":
            schedule = self.freeze_config.get("stage_schedule")
            if schedule:
                ratio = progressive_unfreeze(self.model, epoch=self.current_epoch, schedule=schedule)
                self.log("train/freeze_ratio", ratio, on_epoch=True, prog_bar=False)
                get_logger("ivf").info("Stage phase epoch %s: freeze ratio=%s", self.current_epoch, ratio)
        if self.phase == "quality":
            warmup_epochs = max(self.quality_warmup_epochs, 0)
            if self.current_epoch < warmup_epochs:
                freeze_encoder(self.model, ratio=1.0)
                self._set_trainable(self.model.morph, False)
                if self.use_stage and self.model.stage is not None:
                    self._set_trainable(self.model.stage, False)
                self._set_trainable(self.model.quality, True)
                get_logger("ivf").info(
                    "Quality warmup epoch %s/%s: encoder frozen; quality head trainable.",
                    self.current_epoch + 1,
                    warmup_epochs,
                )
            else:
                self._set_trainable(self.model.morph, False)
                if self.use_stage and self.model.stage is not None:
                    self._set_trainable(self.model.stage, False)
                self._set_trainable(self.model.quality, True)
                if self.quality_unfreeze_last_n_blocks > 0:
                    freeze_encoder(self.model, ratio=1.0)
                    self._unfreeze_last_encoder_blocks(self.quality_unfreeze_last_n_blocks)
                    get_logger("ivf").info(
                        "Quality finetune epoch %s: encoder unfrozen last %s blocks.",
                        self.current_epoch + 1,
                        self.quality_unfreeze_last_n_blocks,
                    )
                elif self.quality_unfreeze_ratio > 0:
                    freeze_ratio = max(0.0, min(1.0, 1.0 - self.quality_unfreeze_ratio))
                    freeze_encoder(self.model, ratio=freeze_ratio)
                    get_logger("ivf").info(
                        "Quality finetune epoch %s: encoder unfreeze_ratio=%.2f (freeze_ratio=%.2f).",
                        self.current_epoch + 1,
                        self.quality_unfreeze_ratio,
                        freeze_ratio,
                    )
                else:
                    freeze_encoder(self.model, ratio=1.0)
                    get_logger("ivf").info(
                        "Quality finetune epoch %s: encoder frozen (no unfreeze config).",
                        self.current_epoch + 1,
                    )
        if not self.trainer or getattr(self.trainer, "sanity_checking", False):
            return
        self._epoch_start_time = time.time()
        self._next_progress_pct = 25.0

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
            self._val_manual_correct = {"icm": 0, "te": 0}
            self._val_manual_total = {"icm": 0, "te": 0}
            self._val_counts = {"exp": 0, "icm": 0, "te": 0}
        if self.phase == "quality":
            self._quality_confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        if self.phase == "q":
            self._q_val_stats = {"count": 0, "sum": 0.0, "sumsq": 0.0, "min": None, "max": None}

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if not self.trainer or getattr(self.trainer, "sanity_checking", False):
            return
        num_batches = getattr(self.trainer, "num_training_batches", None)
        if not isinstance(num_batches, int) or num_batches <= 0:
            return
        if self._epoch_start_time is None:
            self._epoch_start_time = time.time()

        progress = (batch_idx + 1) / num_batches * 100.0
        elapsed = time.time() - self._epoch_start_time
        it_per_s = (batch_idx + 1) / elapsed if elapsed > 0 else 0.0
        is_last = (batch_idx + 1) >= num_batches

        if self.live_epoch_line:
            sys.stdout.write(
                f"\r[epoch {self.current_epoch + 1}] progress={progress:5.1f}% it/s={it_per_s:6.2f}"
            )
            sys.stdout.flush()
            if is_last:
                sys.stdout.write("\n")
                sys.stdout.flush()
            return

        if self._next_progress_pct is None:
            self._next_progress_pct = 25.0
        if progress < self._next_progress_pct and not is_last:
            return

        get_logger("ivf").info(
            "[epoch %s] progress=%.1f%% it/s=%.2f",
            self.current_epoch + 1,
            min(progress, 100.0),
            it_per_s,
        )

        while self._next_progress_pct is not None and progress >= self._next_progress_pct:
            self._next_progress_pct += 25.0

    def configure_optimizers(self):
        if self.phase == "quality" and self.quality_head_lr is not None:
            head_lr = float(self.quality_head_lr)
            encoder_lr = head_lr * self.quality_encoder_lr_scale
            head_params = list(self.model.quality.parameters())
            encoder_params = list(self.model.encoder.parameters())
            return torch.optim.AdamW(
                [
                    {"params": head_params, "lr": head_lr},
                    {"params": encoder_params, "lr": encoder_lr},
                ],
                weight_decay=self.weight_decay,
            )
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

    def _masked_focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor],
        weight: float,
        gamma: float = 2.0,
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
        logits = logits[mask]
        targets = targets[mask]
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        t_log_prob = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        t_prob = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - t_prob) ** gamma) * t_log_prob
        if class_weight is not None:
            class_weight = class_weight.to(logits.device)
            loss = loss * class_weight.gather(0, targets)
        return loss.mean() * weight

    def _masked_regression_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor],
        weight: float,
    ) -> Optional[torch.Tensor]:
        if mask is None:
            mask = targets >= 0
        else:
            mask = mask > 0
        if not mask.any():
            return None
        preds = preds[mask].float()
        targets = targets[mask].float()
        if self.q_loss == "mse":
            loss = F.mse_loss(preds, targets)
        else:
            loss = F.smooth_l1_loss(preds, targets)
        return loss * weight

    def _ranking_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> Optional[torch.Tensor]:
        if preds.numel() < 2 or targets.numel() < 2:
            return None
        num_pairs = self.q_ranking_pairs
        max_items = preds.numel()
        if num_pairs > 0:
            max_items = min(max_items, num_pairs * 2)
        idx = torch.randperm(preds.numel(), device=preds.device)[:max_items]
        if idx.numel() < 2:
            return None
        if idx.numel() % 2 == 1:
            idx = idx[:-1]
        pairs = idx.view(-1, 2)
        left = pairs[:, 0]
        right = pairs[:, 1]
        target_diff = targets[left] - targets[right]
        signs = torch.sign(target_diff)
        keep = signs != 0
        if not keep.any():
            return None
        return F.margin_ranking_loss(
            preds[left][keep],
            preds[right][keep],
            signs[keep],
            margin=self.q_ranking_margin,
        )

    def _compute_losses(self, outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        losses = {}

        if self.phase in {"morph", "joint"}:
            loss_exp = self._masked_ce(
                outputs["morph"]["exp"],
                targets["exp"],
                targets.get("exp_mask"),
                self.loss_weights.get("morph", 1.0),
                class_weight=self.exp_class_weight,
                num_classes=len(EXPANSION_CLASSES),
            )
            if self.use_focal_icm:
                loss_icm = self._masked_focal_loss(
                    outputs["morph"]["icm"],
                    targets["icm"],
                    targets.get("icm_mask"),
                    self.loss_weights.get("morph", 1.0),
                    gamma=self.focal_gamma,
                    class_weight=self.icm_class_weight,
                    num_classes=self.icm_num_classes,
                )
            else:
                loss_icm = self._masked_ce(
                    outputs["morph"]["icm"],
                    targets["icm"],
                    targets.get("icm_mask"),
                    self.loss_weights.get("morph", 1.0),
                    class_weight=self.icm_class_weight,
                    num_classes=self.icm_num_classes,
                )
            if self.use_focal_te:
                loss_te = self._masked_focal_loss(
                    outputs["morph"]["te"],
                    targets["te"],
                    targets.get("te_mask"),
                    self.loss_weights.get("morph", 1.0),
                    gamma=self.focal_gamma,
                    class_weight=self.te_class_weight,
                    num_classes=self.te_num_classes,
                )
            else:
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

        if self.phase in {"stage", "joint"} and self.use_stage:
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

        if self.phase == "q":
            q_loss = self._masked_regression_loss(
                outputs["q"],
                targets["q"],
                targets.get("q_mask"),
                self.loss_weights.get("q", 1.0),
            )
            if q_loss is not None:
                losses["q"] = q_loss

            if self.q_ranking_loss and q_loss is not None:
                q_target = targets["q"].float()
                mask = targets.get("q_mask")
                mask = mask > 0 if mask is not None else q_target >= 0
                if mask.any():
                    q_pred = outputs["q"][mask].float()
                    q_true = q_target[mask]
                    rank_loss = self._ranking_loss(q_pred, q_true)
                    if rank_loss is not None:
                        losses["q_rank"] = rank_loss * self.q_ranking_weight

            if self.q_aux_alpha > 0:
                aux_losses = []
                loss_exp = self._masked_ce(
                    outputs["morph"]["exp"],
                    targets["exp"],
                    targets.get("exp_mask"),
                    self.q_aux_alpha,
                    class_weight=self.exp_class_weight,
                    num_classes=len(EXPANSION_CLASSES),
                )
                if loss_exp is not None:
                    aux_losses.append(loss_exp)
                loss_icm = self._masked_ce(
                    outputs["morph"]["icm"],
                    targets["icm"],
                    targets.get("icm_mask"),
                    self.q_aux_alpha,
                    class_weight=self.icm_class_weight,
                    num_classes=self.icm_num_classes,
                )
                if loss_icm is not None:
                    aux_losses.append(loss_icm)
                loss_te = self._masked_ce(
                    outputs["morph"]["te"],
                    targets["te"],
                    targets.get("te_mask"),
                    self.q_aux_alpha,
                    class_weight=self.te_class_weight,
                    num_classes=self.te_num_classes,
                )
                if loss_te is not None:
                    aux_losses.append(loss_te)
                if self.use_stage and "stage" in outputs:
                    stage_loss = self._masked_ce(
                        outputs["stage"],
                        targets.get("stage", torch.tensor(IGNORE_INDEX, device=outputs["stage"].device)),
                        None,
                        self.q_aux_alpha,
                    )
                    if stage_loss is not None:
                        aux_losses.append(stage_loss)
                if aux_losses:
                    losses["q_aux"] = sum(aux_losses)

        total = sum(losses.values()) if losses else torch.tensor(0.0, device=outputs["features"].device)
        losses["total"] = total
        return losses

    def training_step(self, batch: Dict, batch_idx: int):
        self._guardrails(batch)
        outputs = self.model(batch["image"])
        losses = self._compute_losses(outputs, batch["targets"])
        batch_size = batch["image"].shape[0] if hasattr(batch.get("image"), "shape") else None
        self.log("train/loss", losses["total"], on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        if "morphology" in losses:
            self.log("train/morph_loss", losses["morphology"], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        if "stage" in losses:
            self.log("train/stage_loss", losses["stage"], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        if "quality" in losses:
            self.log("train/quality_loss", losses["quality"], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        if "q" in losses:
            self.log("train/q_loss", losses["q"], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        if "q_aux" in losses:
            self.log("train/q_aux_loss", losses["q_aux"], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        if "q_rank" in losses:
            self.log("train/q_rank_loss", losses["q_rank"], on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        return losses["total"]

    def validation_step(self, batch: Dict, batch_idx: int):
        self._guardrails(batch)
        outputs = self.model(batch["image"])
        losses = self._compute_losses(outputs, batch["targets"])
        batch_size = batch["image"].shape[0] if hasattr(batch.get("image"), "shape") else None
        self.log("val/loss", losses["total"], on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        if "morphology" in losses:
            self.log("val/morph_loss", losses["morphology"], on_epoch=True, prog_bar=False, batch_size=batch_size)
        if "stage" in losses:
            self.log("val/stage_loss", losses["stage"], on_epoch=True, prog_bar=False, batch_size=batch_size)
        if "quality" in losses:
            self.log("val/quality_loss", losses["quality"], on_epoch=True, prog_bar=False, batch_size=batch_size)
        if "q" in losses:
            self.log("val/q_loss", losses["q"], on_epoch=True, prog_bar=False, batch_size=batch_size)
        if "q_aux" in losses:
            self.log("val/q_aux_loss", losses["q_aux"], on_epoch=True, prog_bar=False, batch_size=batch_size)
        if "q_rank" in losses:
            self.log("val/q_rank_loss", losses["q_rank"], on_epoch=True, prog_bar=False, batch_size=batch_size)

        targets = batch["targets"]

        if self.phase in {"morph", "joint"}:
            if "exp_acc" in self.morph_metrics:
                t = targets["exp"]
                mask = targets.get("exp_mask")
                mask = mask > 0 if mask is not None else t >= 0
                mask = mask & (t < len(EXPANSION_CLASSES))
                exp_n = int(mask.sum().item())
                if self._val_counts is not None:
                    self._val_counts["exp"] += exp_n
                if exp_n > 0:
                    preds = outputs["morph"]["exp"].argmax(dim=-1)
                    self.morph_metrics["exp_acc"].update(preds[mask], t[mask])
                    self.log("val/exp_acc", self.morph_metrics["exp_acc"], on_epoch=True, prog_bar=False, batch_size=batch_size)

            for head, num_classes in (("icm", self.icm_num_classes), ("te", self.te_num_classes)):
                t = targets[head]
                mask = targets.get(f"{head}_mask")
                mask = mask > 0 if mask is not None else t >= 0
                mask = mask & (t < num_classes)
                head_n = int(mask.sum().item())
                if self._val_counts is not None:
                    self._val_counts[head] += head_n
                if head_n > 0:
                    logits = outputs["morph"][head][:, :num_classes]
                    preds = logits.argmax(dim=-1)
                    for metric_key in (f"{head}_acc", f"{head}_bal_acc", f"{head}_macro_f1"):
                        if metric_key in self.morph_metrics:
                            metric = self.morph_metrics[metric_key]
                            metric.update(preds[mask], t[mask])
                            self.log(f"val/{metric_key}", metric, on_epoch=True, prog_bar=False, batch_size=batch_size)
                    correct = (preds[mask] == t[mask]).sum().item()
                    total = int(mask.sum().item())
                    if self._val_manual_correct is not None:
                        self._val_manual_correct[head] += int(correct)
                    if self._val_manual_total is not None:
                        self._val_manual_total[head] += int(total)

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

        if self.phase in {"stage", "joint"} and self.use_stage:
            t = targets["stage"]
            mask = t >= 0
            if mask.any():
                preds = outputs["stage"].argmax(dim=-1)
                for key, metric in self.stage_metrics.items():
                    metric.update(preds[mask], t[mask])
                    self.log(f"val/{key}", metric, on_epoch=True, prog_bar=False, batch_size=batch_size)

        if self.phase == "quality":
            t = targets["quality"]
            mask = t >= 0
            if mask.any():
                logits = outputs["quality"][:, 1] - outputs["quality"][:, 0]
                probs = torch.sigmoid(logits)
                for key, metric in self.quality_metrics.items():
                    metric.update(probs[mask], t[mask])
                    self.log(f"val/{key}", metric, on_epoch=True, prog_bar=False, batch_size=batch_size)
                preds = (probs >= 0.5).long()
                y_true = t.long()
                if self._quality_confusion is not None:
                    tp = int(((preds == 1) & (y_true == 1) & mask).sum().item())
                    tn = int(((preds == 0) & (y_true == 0) & mask).sum().item())
                    fp = int(((preds == 1) & (y_true == 0) & mask).sum().item())
                    fn = int(((preds == 0) & (y_true == 1) & mask).sum().item())
                    self._quality_confusion["tp"] += tp
                    self._quality_confusion["tn"] += tn
                    self._quality_confusion["fp"] += fp
                    self._quality_confusion["fn"] += fn

        if self.phase == "q":
            q_target = targets["q"].float()
            mask = targets.get("q_mask")
            mask = mask > 0 if mask is not None else q_target >= 0
            q_n = int(mask.sum().item())
            if q_n > 0:
                q_pred = outputs["q"][mask].float()
                q_true = q_target[mask]
                rmse = torch.sqrt(torch.mean((q_pred - q_true) ** 2))
                mae = torch.mean(torch.abs(q_pred - q_true))
                self.log("val/q_rmse", rmse, on_epoch=True, prog_bar=False, batch_size=batch_size)
                self.log("val/q_mae", mae, on_epoch=True, prog_bar=False, batch_size=batch_size)
                self.log("val/q_n", q_n, on_epoch=True, prog_bar=False, batch_size=batch_size)
                if self._q_val_stats is not None:
                    q_pred_cpu = q_pred.detach().cpu()
                    self._q_val_stats["count"] += int(q_pred_cpu.numel())
                    self._q_val_stats["sum"] += float(q_pred_cpu.sum().item())
                    self._q_val_stats["sumsq"] += float((q_pred_cpu ** 2).sum().item())
                    current_min = float(q_pred_cpu.min().item())
                    current_max = float(q_pred_cpu.max().item())
                    if self._q_val_stats["min"] is None or current_min < self._q_val_stats["min"]:
                        self._q_val_stats["min"] = current_min
                    if self._q_val_stats["max"] is None or current_max > self._q_val_stats["max"]:
                        self._q_val_stats["max"] = current_max
            else:
                device = getattr(self, "device", None) or q_target.device
                self.log("val/q_rmse", torch.tensor(float("nan"), device=device), on_epoch=True, prog_bar=False, batch_size=batch_size)
                self.log("val/q_mae", torch.tensor(float("nan"), device=device), on_epoch=True, prog_bar=False, batch_size=batch_size)
                self.log("val/q_n", 0, on_epoch=True, prog_bar=False, batch_size=batch_size)

    def on_validation_epoch_end(self) -> None:
        if self.trainer and getattr(self.trainer, "sanity_checking", False):
            return

        metrics = self.trainer.callback_metrics if self.trainer else {}
        quality_bal_acc = None
        if self.phase == "quality" and self._quality_confusion is not None:
            tp = self._quality_confusion["tp"]
            tn = self._quality_confusion["tn"]
            fp = self._quality_confusion["fp"]
            fn = self._quality_confusion["fn"]
            tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            tnr = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
            if tpr == tpr and tnr == tnr:
                quality_bal_acc = 0.5 * (tpr + tnr)
                device = getattr(self, "device", None) or torch.device("cpu")
                self.log("val/quality_bal_acc", torch.tensor(quality_bal_acc, device=device), on_epoch=True, prog_bar=False)

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
            if quality_bal_acc is not None:
                parts.append(f"val_bal_acc={quality_bal_acc:.4f}")
        if self.phase == "q":
            _append(parts, "val_q_rmse", "val/q_rmse")
            _append(parts, "val_q_mae", "val/q_mae")
            if self._q_val_stats is not None and self._q_val_stats.get("count", 0) > 0:
                count = self._q_val_stats["count"]
                mean = self._q_val_stats["sum"] / count
                var = max(self._q_val_stats["sumsq"] / count - mean ** 2, 0.0)
                std = var ** 0.5
                device = getattr(self, "device", None) or torch.device("cpu")
                self.log("val/q_mean", torch.tensor(mean, device=device), on_epoch=True, prog_bar=False)
                self.log("val/q_std", torch.tensor(std, device=device), on_epoch=True, prog_bar=False)
                self.log("val/q_min", torch.tensor(self._q_val_stats["min"], device=device), on_epoch=True, prog_bar=False)
                self.log("val/q_max", torch.tensor(self._q_val_stats["max"], device=device), on_epoch=True, prog_bar=False)
                parts.append(f"val_q_mean={mean:.4f}")
                parts.append(f"val_q_std={std:.4f}")

        if len(parts) > 1:
            get_logger("ivf").info(" ".join(parts))

        if self._val_counts and self.phase in {"morph", "joint"}:
            self.log("val_exp_n", self._val_counts.get("exp", 0), on_epoch=True, prog_bar=False)
            self.log("val_icm_n", self._val_counts.get("icm", 0), on_epoch=True, prog_bar=False)
            self.log("val_te_n", self._val_counts.get("te", 0), on_epoch=True, prog_bar=False)
            get_logger("ivf").info(
                "Validation labeled counts: exp=%s icm=%s te=%s",
                self._val_counts.get("exp", 0),
                self._val_counts.get("icm", 0),
                self._val_counts.get("te", 0),
            )
            device = getattr(self, "device", None) or torch.device("cpu")
            if self._val_counts.get("exp", 0) == 0:
                self.log("val/exp_acc", torch.tensor(float("nan"), device=device), on_epoch=True, prog_bar=False)
            if self._val_counts.get("icm", 0) == 0:
                for key in ("val/icm_acc", "val/icm_bal_acc", "val/icm_macro_f1"):
                    self.log(key, torch.tensor(float("nan"), device=device), on_epoch=True, prog_bar=False)
            if self._val_counts.get("te", 0) == 0:
                for key in ("val/te_acc", "val/te_bal_acc", "val/te_macro_f1"):
                    self.log(key, torch.tensor(float("nan"), device=device), on_epoch=True, prog_bar=False)

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

                total = int(counts.sum().item())
                head_n = self._val_counts.get(head, 0) if self._val_counts else 0
                if head_n == 0:
                    logger.info("Validation %s has no labeled samples; collapse check skipped.", head)
                    continue
                if head_n < 20:
                    logger.warning(
                        "Validation %s labeled samples=%s < 20; collapse check skipped.",
                        head,
                        head_n,
                    )
                    continue
                if total <= 0 or true_counts is None:
                    continue
                pred_unique = int((counts > 0).sum().item())
                true_unique = int((true_counts > 0).sum().item())
                majority_freq = float(counts.max().item() / total) if total > 0 else 0.0
                collapse = pred_unique == 1 and true_unique >= 2 and majority_freq >= 0.95
                if collapse:
                    self._collapse_streak[head] = self._collapse_streak.get(head, 0) + 1
                    if self._collapse_streak[head] >= 3:
                        logger.warning(
                            "Prediction collapse detected for %s head (streak=%s, majority=%.2f, pred_unique=%s, true_unique=%s).",
                            head,
                            self._collapse_streak[head],
                            majority_freq,
                            pred_unique,
                            true_unique,
                        )
                    else:
                        logger.info(
                            "Prediction collapse signal for %s head (streak=%s, majority=%.2f, pred_unique=%s, true_unique=%s).",
                            head,
                            self._collapse_streak[head],
                            majority_freq,
                            pred_unique,
                            true_unique,
                        )
                else:
                    self._collapse_streak[head] = 0

            if self._val_manual_correct and self._val_manual_total:
                for head in ("icm", "te"):
                    if f"{head}_acc" in self.morph_metrics:
                        continue
                    total = self._val_manual_total.get(head, 0)
                    if total > 0:
                        acc = self._val_manual_correct.get(head, 0) / total
                        self.log(f"val/{head}_acc", acc, on_epoch=True, prog_bar=False)
                        logger.info("Validation %s manual acc: %.4f", head, acc)

        for metric in list(self.morph_metrics.values()) + list(self.stage_metrics.values()) + list(self.quality_metrics.values()):
            metric.reset()
        if self.phase == "quality":
            self._quality_confusion = None

    def _setup_morph_class_weights(self) -> None:
        if not self.trainer or not hasattr(self.trainer, "datamodule"):
            return
        dataset = getattr(self.trainer.datamodule, "train_dataset", None)
        if dataset is None:
            return

        counts = {
            "exp": torch.zeros(len(EXPANSION_CLASSES), dtype=torch.long),
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
            exp_label = targets.get("exp", IGNORE_INDEX)
            exp_mask = targets.get("exp_mask", 0)
            if exp_mask and exp_label is not None and exp_label >= 0:
                if exp_label < counts["exp"].numel():
                    counts["exp"][int(exp_label)] += 1
            for head in ("icm", "te"):
                label = targets.get(head, IGNORE_INDEX)
                mask = targets.get(f"{head}_mask", 0)
                if mask and label is not None and label >= 0:
                    if label < counts[head].numel():
                        counts[head][int(label)] += 1

        logger = get_logger("ivf")
        logger.info(
            "Morph train exp counts: %s",
            {exp: int(counts["exp"][i]) for i, exp in enumerate(EXPANSION_CLASSES)},
        )
        logger.info("Morph train icm counts: %s", {cls: int(counts["icm"][i]) for i, cls in enumerate(ICM_CLASSES)})
        logger.info("Morph train te counts: %s", {cls: int(counts["te"][i]) for i, cls in enumerate(TE_CLASSES)})
        self.exp_class_counts = counts["exp"].clone()
        self.icm_class_counts = counts["icm"].clone()
        self.te_class_counts = counts["te"].clone()

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

        def _imbalance_ratio(head_counts: torch.Tensor, num_classes: int) -> Optional[float]:
            if num_classes <= 0:
                return None
            counts_slice = head_counts[:num_classes].float()
            if counts_slice.numel() == 0:
                return None
            min_count = counts_slice.min().item()
            max_count = counts_slice.max().item()
            if min_count <= 0:
                return float("inf")
            return max_count / min_count

        def _compute_weights(head: str, num_classes: int) -> Optional[torch.Tensor]:
            head_counts = counts[head][:num_classes].float()
            total = head_counts.sum().item()
            if total <= 0:
                return None
            weights = torch.ones_like(head_counts)
            for i, c in enumerate(head_counts):
                if c > 0:
                    weights[i] = total / (num_classes * c)
                else:
                    weights[i] = 0.0
            return weights

        if self.use_class_weights:
            self.exp_class_weight = _compute_weights("exp", len(EXPANSION_CLASSES))
            self.icm_class_weight = _compute_weights("icm", self.icm_num_classes)
            self.te_class_weight = _compute_weights("te", self.te_num_classes)
            if self.exp_class_weight is not None:
                logger.info("EXP class weights: %s", self.exp_class_weight.tolist())
            if self.icm_class_weight is not None:
                logger.info("ICM class weights: %s", self.icm_class_weight.tolist())
            if self.te_class_weight is not None:
                logger.info("TE class weights: %s", self.te_class_weight.tolist())
        else:
            self.exp_class_weight = None
            self.icm_class_weight = None
            self.te_class_weight = None
            logger.info("Morph class weights disabled by config.")

        icm_ratio = _imbalance_ratio(counts["icm"], self.icm_num_classes)
        te_ratio = _imbalance_ratio(counts["te"], self.te_num_classes)
        self.use_focal_icm = icm_ratio is not None and icm_ratio > 3.0
        self.use_focal_te = te_ratio is not None and te_ratio > 3.0
        logger.info(
            "Morph icm imbalance ratio=%s focal=%s gamma=%s",
            "n/a" if icm_ratio is None else ("inf" if icm_ratio == float("inf") else f"{icm_ratio:.2f}"),
            "on" if self.use_focal_icm else "off",
            self.focal_gamma,
        )
        logger.info(
            "Morph te imbalance ratio=%s focal=%s gamma=%s",
            "n/a" if te_ratio is None else ("inf" if te_ratio == float("inf") else f"{te_ratio:.2f}"),
            "on" if self.use_focal_te else "off",
            self.focal_gamma,
        )

        self.morph_metrics["icm_acc"] = MulticlassAccuracy(num_classes=self.icm_num_classes)
        self.morph_metrics["icm_bal_acc"] = MulticlassAccuracy(num_classes=self.icm_num_classes, average="macro")
        self.morph_metrics["icm_macro_f1"] = MulticlassF1Score(num_classes=self.icm_num_classes, average="macro")
        self.morph_metrics["te_acc"] = MulticlassAccuracy(num_classes=self.te_num_classes)
        self.morph_metrics["te_bal_acc"] = MulticlassAccuracy(num_classes=self.te_num_classes, average="macro")
        self.morph_metrics["te_macro_f1"] = MulticlassF1Score(num_classes=self.te_num_classes, average="macro")
        device = getattr(self, "device", None)
        if device is not None:
            for key in ("icm_acc", "icm_bal_acc", "icm_macro_f1", "te_acc", "te_bal_acc", "te_macro_f1"):
                self.morph_metrics[key] = self.morph_metrics[key].to(device)
