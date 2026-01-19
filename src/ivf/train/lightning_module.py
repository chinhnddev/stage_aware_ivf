"""
Lightning module for multi-phase IVF training.
"""

import math
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


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {}
        for k, v in model.state_dict().items():
            if torch.is_floating_point(v) or torch.is_complex(v):
                self.shadow[k] = v.detach().clone()

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k not in self.shadow:
                    continue
                if not (torch.is_floating_point(v) or torch.is_complex(v)):
                    continue
                shadow = self.shadow[k]
                value = v.detach()
                if value.dtype != shadow.dtype:
                    value = value.to(dtype=shadow.dtype)
                shadow.mul_(self.decay).add_(value, alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if self.shadow:
            model.load_state_dict(self.shadow, strict=False)
        return backup

    def restore(self, model: nn.Module, backup) -> None:
        if backup:
            model.load_state_dict(backup, strict=False)


class MultiTaskLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        phase: str,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        use_cosine_warmup: bool = False,
        warmup_epochs: int = 5,
        min_lr: float = 0.0,
        ema_decay: float = 0.0,
        loss_weights: Optional[Dict[str, float]] = None,
        freeze_config: Optional[Dict] = None,
        morph_loss_reduction: str = "mean",
        morph_mode: str = "multi_task",
        single_task_head: Optional[str] = None,
        mtl_grad_strategy: str = "none",
        exp_num_classes: Optional[int] = None,
        morph_lambda_icm: float = 1.0,
        morph_lambda_te: float = 1.0,
        quality_pos_weight: Optional[float] = None,
        use_class_weights: bool = False,
        class_weight_mode: str = "inverse_freq",
        morph_class_weights_exp: Optional[list] = None,
        morph_class_weights_icm: Optional[list] = None,
        morph_class_weights_te: Optional[list] = None,
        q_loss: str = "smoothl1",
        q_aux_alpha: float = 0.0,
        q_freeze_backbone: bool = True,
        use_focal_icm: bool = False,
        use_focal_te: bool = False,
        focal_gamma: float = 2.0,
        live_epoch_line: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.phase = phase
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_cosine_warmup = use_cosine_warmup
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.ema_decay = ema_decay
        self.ema = EMA(self.model, decay=ema_decay) if ema_decay > 0 else None
        self._ema_backup = None
        self.loss_weights = loss_weights or {"morph": 1.0, "stage": 1.0, "quality": 1.0}
        if self.phase == "morph":
            self.loss_weights = dict(self.loss_weights)
            self.loss_weights["stage"] = 0.0
            self.loss_weights["quality"] = 0.0
            get_logger("ivf").info("Morph phase: forcing stage/quality loss weights to 0.")
        self.freeze_config = freeze_config or {}
        self.morph_loss_reduction = morph_loss_reduction
        self.morph_mode = morph_mode
        self.single_task_head = single_task_head
        self.mtl_grad_strategy = mtl_grad_strategy
        self.morph_lambda_icm = morph_lambda_icm
        self.morph_lambda_te = morph_lambda_te
        self.quality_pos_weight = quality_pos_weight
        self.use_class_weights = use_class_weights
        self.class_weight_mode = class_weight_mode
        self.morph_class_weights_exp = morph_class_weights_exp
        self.morph_class_weights_icm = morph_class_weights_icm
        self.morph_class_weights_te = morph_class_weights_te
        self.q_loss = q_loss
        self.q_aux_alpha = q_aux_alpha
        self.q_freeze_backbone = q_freeze_backbone
        self.live_epoch_line = live_epoch_line
        self._epoch_start_time = None
        self._val_pred_counts = None
        self._val_true_counts = None
        self._val_manual_correct = None
        self._val_manual_total = None
        self._val_counts = None
        self._collapse_streak = {"icm": 0, "te": 0}
        self.exp_class_weight = None
        self.icm_class_weight = None
        self.te_class_weight = None
        self.exp_num_classes = exp_num_classes or len(EXPANSION_CLASSES)
        self.icm_num_classes = len(ICM_CLASSES)
        self.te_num_classes = len(TE_CLASSES)
        self.exp_class_counts = None
        self.icm_class_counts = None
        self.te_class_counts = None
        self.focal_gamma = focal_gamma
        self.use_focal_icm = use_focal_icm
        self.use_focal_te = use_focal_te

        if self.morph_loss_reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported morph_loss_reduction: {self.morph_loss_reduction}")
        if self.morph_mode not in {"single_task", "multi_task"}:
            raise ValueError(f"Unsupported morph_mode: {self.morph_mode}")
        if self.morph_mode == "single_task":
            if self.single_task_head not in {"exp", "icm", "te"}:
                raise ValueError("single_task_head must be one of exp, icm, te when morph_mode=single_task.")
        else:
            self.single_task_head = None
        if self.mtl_grad_strategy not in {"none", "pcgrad", "gradnorm"}:
            raise ValueError(f"Unsupported mtl_grad_strategy: {self.mtl_grad_strategy}")
        if self.mtl_grad_strategy == "gradnorm":
            get_logger("ivf").warning("GradNorm not implemented; falling back to mtl_grad_strategy=none.")
            self.mtl_grad_strategy = "none"
        if self.class_weight_mode not in {"inverse_freq"}:
            raise ValueError(f"Unsupported class_weight_mode: {self.class_weight_mode}")
        if self.q_loss not in {"smoothl1", "mse"}:
            raise ValueError(f"Unsupported q_loss: {self.q_loss}")

        self._apply_phase_freeze(initial=True)

        self.morph_metrics = nn.ModuleDict(build_morphology_metrics(exp_num_classes=self.exp_num_classes))
        self.stage_metrics = nn.ModuleDict(build_stage_metrics())
        self.quality_metrics = nn.ModuleDict(build_quality_metrics())

        self._epoch_start_time = None
        self._next_progress_pct = None

        self.automatic_optimization = not (
            self.phase == "morph"
            and self.morph_mode == "multi_task"
            and self.mtl_grad_strategy == "pcgrad"
        )

        self.save_hyperparameters(ignore=["model"])

    def _apply_phase_freeze(self, initial: bool = False) -> None:
        logger = get_logger("ivf")
        def _set_trainable(module: nn.Module, trainable: bool) -> None:
            for p in module.parameters():
                p.requires_grad = trainable

        def _set_trainable_if_exists(name: str, trainable: bool) -> None:
            module = getattr(self.model, name, None)
            if module is not None:
                _set_trainable(module, trainable)

        def _unfreeze_last_encoder_blocks(n_blocks: int) -> None:
            if n_blocks <= 0:
                return
            encoder = getattr(self.model, "encoder", None)
            blocks = getattr(encoder, "blocks", None)
            if blocks is None:
                logger.warning("Encoder blocks not found; q_unfreeze_last_n_blocks ignored.")
                return
            for block in list(blocks)[-n_blocks:]:
                _set_trainable(block, True)
            if hasattr(encoder, "proj"):
                _set_trainable(encoder.proj, True)

        if self.phase == "morph":
            _set_trainable_if_exists("encoder", True)
            _set_trainable_if_exists("morph", True)
            _set_trainable_if_exists("stage", False)
            _set_trainable_if_exists("quality", False)
            logger.info("EXP-1 Morphology: encoder+morph trainable; stage+quality frozen.")
        elif self.phase == "stage":
            _set_trainable_if_exists("encoder", True)
            _set_trainable_if_exists("morph", False)
            _set_trainable_if_exists("stage", True)
            _set_trainable_if_exists("quality", False)
            if initial:
                freeze_ratio = self.freeze_config.get("stage_start_ratio", 0.8)
                freeze_encoder(self.model, ratio=freeze_ratio)
                logger.info("EXP-2 Stage-aware: initial freeze ratio=%s", freeze_ratio)
        elif self.phase == "joint":
            _set_trainable_if_exists("encoder", True)
            _set_trainable_if_exists("morph", True)
            _set_trainable_if_exists("stage", True)
            _set_trainable_if_exists("quality", False)
            logger.info("EXP-3 Joint stabilization: encoder+morph+stage trainable; quality frozen.")
        elif self.phase == "quality":
            freeze_encoder(self.model, ratio=1.0)
            _set_trainable_if_exists("morph", False)
            _set_trainable_if_exists("stage", False)
            _set_trainable_if_exists("quality", True)
            logger.info("EXP-4 Quality: encoder+morph+stage frozen; quality trainable.")
        elif self.phase == "q":
            if self.q_freeze_backbone:
                freeze_encoder(self.model, ratio=1.0)
            else:
                _set_trainable_if_exists("encoder", True)
            _set_trainable_if_exists("morph", False)
            _set_trainable_if_exists("stage", False)
            _set_trainable_if_exists("quality", False)
            _set_trainable_if_exists("q_head", True)
            unfreeze_blocks = int(self.freeze_config.get("q_unfreeze_last_n_blocks", 0))
            if self.q_freeze_backbone and unfreeze_blocks > 0:
                _unfreeze_last_encoder_blocks(unfreeze_blocks)
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
        if not self.trainer or getattr(self.trainer, "sanity_checking", False):
            return
        self._epoch_start_time = time.time()
        self._next_progress_pct = 25.0

    def on_fit_start(self) -> None:
        logger = get_logger("ivf")
        if self.use_cosine_warmup:
            logger.info("LR schedule: cosine+warmup warmup_epochs=%s min_lr=%s", self.warmup_epochs, self.min_lr)
        if self.ema is not None:
            logger.info("EMA enabled: decay=%s", self.ema_decay)
        if self.phase not in {"morph", "joint"}:
            return
        if self.icm_class_weight is not None or self.te_class_weight is not None:
            return
        self._setup_morph_class_weights()

    def on_validation_epoch_start(self) -> None:
        if self.ema is not None and self.trainer and not getattr(self.trainer, "sanity_checking", False):
            self._ema_backup = self.ema.apply_to(self.model)
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

        if self.ema is not None:
            self.ema.update(self.model)

    def _estimate_total_steps(self) -> Optional[int]:
        if not self.trainer:
            return None
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if total_steps is None:
            num_batches = getattr(self.trainer, "num_training_batches", None)
            max_epochs = getattr(self.trainer, "max_epochs", None)
            if isinstance(num_batches, int) and isinstance(max_epochs, int):
                total_steps = num_batches * max_epochs
        if total_steps is None:
            return None
        try:
            return int(total_steps)
        except (TypeError, ValueError):
            return None

    def _build_lr_scheduler(self, optimizer):
        total_steps = self._estimate_total_steps()
        if not total_steps or total_steps <= 0:
            return None
        max_epochs = getattr(self.trainer, "max_epochs", 1) if self.trainer else 1
        steps_per_epoch = total_steps / max(1, max_epochs)
        warmup_steps = int(self.warmup_epochs * steps_per_epoch)
        base_lr = float(self.lr)
        min_lr = float(self.min_lr)
        min_ratio = min_lr / base_lr if base_lr > 0 else 0.0

        def _lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return min_ratio + (1.0 - min_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    def _step_schedulers(self) -> None:
        schedulers = self.lr_schedulers()
        if schedulers is None:
            return
        if isinstance(schedulers, (list, tuple)):
            for scheduler in schedulers:
                scheduler.step()
        else:
            schedulers.step()

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        if not self.use_cosine_warmup:
            return optimizer
        scheduler = self._build_lr_scheduler(optimizer)
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

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

    def _is_head_active(self, head: str) -> bool:
        if self.morph_mode != "single_task":
            return True
        return self.single_task_head == head

    def _compute_morph_task_losses(self, outputs: Dict, targets: Dict) -> Dict[str, Optional[torch.Tensor]]:
        losses = {"exp": None, "icm": None, "te": None}
        base_weight = float(self.loss_weights.get("morph", 1.0))
        if not self._is_head_active("exp"):
            pass
        else:
            losses["exp"] = self._masked_ce(
                outputs["morph"]["exp"],
                targets["exp"],
                targets.get("exp_mask"),
                base_weight,
                class_weight=self.exp_class_weight,
                num_classes=self.exp_num_classes,
            )

        if self._is_head_active("icm"):
            if self.use_focal_icm:
                losses["icm"] = self._masked_focal_loss(
                    outputs["morph"]["icm"],
                    targets["icm"],
                    targets.get("icm_mask"),
                    base_weight * float(self.morph_lambda_icm),
                    gamma=self.focal_gamma,
                    class_weight=self.icm_class_weight,
                    num_classes=self.icm_num_classes,
                )
            else:
                losses["icm"] = self._masked_ce(
                    outputs["morph"]["icm"],
                    targets["icm"],
                    targets.get("icm_mask"),
                    base_weight * float(self.morph_lambda_icm),
                    class_weight=self.icm_class_weight,
                    num_classes=self.icm_num_classes,
                )

        if self._is_head_active("te"):
            if self.use_focal_te:
                losses["te"] = self._masked_focal_loss(
                    outputs["morph"]["te"],
                    targets["te"],
                    targets.get("te_mask"),
                    base_weight * float(self.morph_lambda_te),
                    gamma=self.focal_gamma,
                    class_weight=self.te_class_weight,
                    num_classes=self.te_num_classes,
                )
            else:
                losses["te"] = self._masked_ce(
                    outputs["morph"]["te"],
                    targets["te"],
                    targets.get("te_mask"),
                    base_weight * float(self.morph_lambda_te),
                    class_weight=self.te_class_weight,
                    num_classes=self.te_num_classes,
                )

        return losses

    def _reduce_morph_losses(self, task_losses: Dict[str, Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
        morph_losses = [loss for loss in task_losses.values() if loss is not None]
        if not morph_losses:
            return None
        total = sum(morph_losses)
        if self.morph_loss_reduction == "mean":
            total = total / len(morph_losses)
        return total

    def _use_pcgrad(self) -> bool:
        return (
            self.phase == "morph"
            and self.morph_mode == "multi_task"
            and self.mtl_grad_strategy == "pcgrad"
        )

    def _pcgrad_update(self, task_losses, optimizer) -> None:
        params = [p for p in self.parameters() if p.requires_grad]
        grads = []
        for loss in task_losses:
            grad = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            grads.append([g.detach().clone() if g is not None else None for g in grad])

        for i in range(len(grads)):
            for j in range(len(grads)):
                if i == j:
                    continue
                dot = None
                norm = None
                for g_i, g_j in zip(grads[i], grads[j]):
                    if g_i is None or g_j is None:
                        continue
                    dot_val = (g_i * g_j).sum()
                    norm_val = (g_j * g_j).sum()
                    dot = dot_val if dot is None else dot + dot_val
                    norm = norm_val if norm is None else norm + norm_val
                if dot is None or norm is None:
                    continue
                if dot < 0 and norm > 0:
                    coeff = dot / norm
                    for idx, (g_i, g_j) in enumerate(zip(grads[i], grads[j])):
                        if g_i is None or g_j is None:
                            continue
                        grads[i][idx] = g_i - coeff * g_j

        for p_idx, p in enumerate(params):
            grad_sum = None
            for g in grads:
                g_i = g[p_idx]
                if g_i is None:
                    continue
                grad_sum = g_i if grad_sum is None else grad_sum + g_i
            if grad_sum is not None:
                p.grad = grad_sum
        optimizer.step()

    def _compute_losses(self, outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        losses = {}

        if self.phase in {"morph", "joint"}:
            task_losses = self._compute_morph_task_losses(outputs, targets)
            total = self._reduce_morph_losses(task_losses)
            if total is not None:
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

        if self.phase == "q":
            q_loss = self._masked_regression_loss(
                outputs["q"],
                targets["q"],
                targets.get("q_mask"),
                self.loss_weights.get("q", 1.0),
            )
            if q_loss is not None:
                losses["q"] = q_loss

            if self.q_aux_alpha > 0:
                aux_losses = []
                loss_exp = self._masked_ce(
                    outputs["morph"]["exp"],
                    targets["exp"],
                    targets.get("exp_mask"),
                    self.q_aux_alpha,
                    class_weight=self.exp_class_weight,
                    num_classes=self.exp_num_classes,
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
        if self._use_pcgrad():
            optimizer = self.optimizers()
            optimizer.zero_grad()
            task_losses = self._compute_morph_task_losses(outputs, batch["targets"])
            total = self._reduce_morph_losses(task_losses)
            if total is None:
                total = torch.tensor(0.0, device=outputs["features"].device)
            active_losses = [loss for loss in task_losses.values() if loss is not None]
            if active_losses:
                if len(active_losses) == 1:
                    self.manual_backward(active_losses[0])
                    optimizer.step()
                else:
                    self._pcgrad_update(active_losses, optimizer)
                if self.use_cosine_warmup:
                    self._step_schedulers()
            losses = {"total": total, "morphology": total}
        else:
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

        targets = batch["targets"]

        if self.phase in {"morph", "joint"}:
            if self._is_head_active("exp") and "exp_acc" in self.morph_metrics:
                t = targets["exp"]
                mask = targets.get("exp_mask")
                mask = mask > 0 if mask is not None else t >= 0
                mask = mask & (t < self.exp_num_classes)
                exp_n = int(mask.sum().item())
                if self._val_counts is not None:
                    self._val_counts["exp"] += exp_n
                if exp_n > 0:
                    preds = outputs["morph"]["exp"][:, : self.exp_num_classes].argmax(dim=-1)
                    self.morph_metrics["exp_acc"].update(preds[mask], t[mask])
                    self.log("val/exp_acc", self.morph_metrics["exp_acc"], on_epoch=True, prog_bar=False, batch_size=batch_size)
                    if "exp_macro_f1" in self.morph_metrics:
                        self.morph_metrics["exp_macro_f1"].update(preds[mask], t[mask])
                        self.log(
                            "val/exp_macro_f1",
                            self.morph_metrics["exp_macro_f1"],
                            on_epoch=True,
                            prog_bar=False,
                            batch_size=batch_size,
                        )

            for head, num_classes in (("icm", self.icm_num_classes), ("te", self.te_num_classes)):
                if not self._is_head_active(head):
                    continue
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
                    if not self._is_head_active(head):
                        continue
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
            else:
                device = getattr(self, "device", None) or q_target.device
                self.log("val/q_rmse", torch.tensor(float("nan"), device=device), on_epoch=True, prog_bar=False, batch_size=batch_size)
                self.log("val/q_mae", torch.tensor(float("nan"), device=device), on_epoch=True, prog_bar=False, batch_size=batch_size)
                self.log("val/q_n", 0, on_epoch=True, prog_bar=False, batch_size=batch_size)

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
            if self._is_head_active("exp"):
                _append(parts, "val_exp_acc", "val/exp_acc")
                _append(parts, "val_exp_f1", "val/exp_macro_f1")
            if self._is_head_active("icm"):
                _append(parts, "val_icm_acc", "val/icm_acc")
            if self._is_head_active("te"):
                _append(parts, "val_te_acc", "val/te_acc")
        if self.phase in {"stage", "joint"}:
            _append(parts, "val_stage_acc", "val/stage_acc")
            _append(parts, "val_stage_f1", "val/stage_f1")
        if self.phase == "quality":
            _append(parts, "val_auroc", "val/quality_auroc")
            _append(parts, "val_auprc", "val/quality_auprc")
            _append(parts, "val_f1", "val/quality_f1")
            _append(parts, "val_acc", "val/quality_acc")
        if self.phase == "q":
            _append(parts, "val_q_rmse", "val/q_rmse")
            _append(parts, "val_q_mae", "val/q_mae")

        if len(parts) > 1:
            get_logger("ivf").info(" ".join(parts))

        if self._val_counts and self.phase in {"morph", "joint"}:
            if self._is_head_active("exp"):
                self.log("val_exp_n", self._val_counts.get("exp", 0), on_epoch=True, prog_bar=False)
            if self._is_head_active("icm"):
                self.log("val_icm_n", self._val_counts.get("icm", 0), on_epoch=True, prog_bar=False)
            if self._is_head_active("te"):
                self.log("val_te_n", self._val_counts.get("te", 0), on_epoch=True, prog_bar=False)
            get_logger("ivf").info(
                "Validation labeled counts: exp=%s icm=%s te=%s",
                self._val_counts.get("exp", 0),
                self._val_counts.get("icm", 0),
                self._val_counts.get("te", 0),
            )
            device = getattr(self, "device", None) or torch.device("cpu")
            if self._is_head_active("exp") and self._val_counts.get("exp", 0) == 0:
                self.log("val/exp_acc", torch.tensor(float("nan"), device=device), on_epoch=True, prog_bar=False)
            if self._is_head_active("icm") and self._val_counts.get("icm", 0) == 0:
                for key in ("val/icm_acc", "val/icm_bal_acc", "val/icm_macro_f1"):
                    self.log(key, torch.tensor(float("nan"), device=device), on_epoch=True, prog_bar=False)
            if self._is_head_active("te") and self._val_counts.get("te", 0) == 0:
                for key in ("val/te_acc", "val/te_bal_acc", "val/te_macro_f1"):
                    self.log(key, torch.tensor(float("nan"), device=device), on_epoch=True, prog_bar=False)

        if self._val_pred_counts and self.phase in {"morph", "joint"}:
            logger = get_logger("ivf")
            for head, classes in (("icm", ICM_CLASSES[: self.icm_num_classes]), ("te", TE_CLASSES[: self.te_num_classes])):
                if not self._is_head_active(head):
                    continue
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
                    if not self._is_head_active(head):
                        continue
                    if f"{head}_acc" in self.morph_metrics:
                        continue
                    total = self._val_manual_total.get(head, 0)
                    if total > 0:
                        acc = self._val_manual_correct.get(head, 0) / total
                        self.log(f"val/{head}_acc", acc, on_epoch=True, prog_bar=False)
                        logger.info("Validation %s manual acc: %.4f", head, acc)

        for metric in list(self.morph_metrics.values()) + list(self.stage_metrics.values()) + list(self.quality_metrics.values()):
            metric.reset()

        if self._ema_backup is not None and self.ema is not None:
            self.ema.restore(self.model, self._ema_backup)
            self._ema_backup = None

    def on_save_checkpoint(self, checkpoint: Dict) -> None:
        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.shadow

    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        if self.ema is None:
            return
        ema_state = checkpoint.get("ema_state_dict")
        if ema_state is not None:
            self.ema.shadow = ema_state

    def _setup_morph_class_weights(self) -> None:
        if not self.trainer or not hasattr(self.trainer, "datamodule"):
            return
        dataset = getattr(self.trainer.datamodule, "train_dataset", None)
        if dataset is None:
            return

        counts = {
            "exp": torch.zeros(self.exp_num_classes, dtype=torch.long),
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
            {exp: int(counts["exp"][i]) for i, exp in enumerate(range(1, self.exp_num_classes + 1))},
        )
        logger.info("Morph train icm counts: %s", {cls: int(counts["icm"][i]) for i, cls in enumerate(ICM_CLASSES)})
        logger.info("Morph train te counts: %s", {cls: int(counts["te"][i]) for i, cls in enumerate(TE_CLASSES)})
        self.exp_class_counts = counts["exp"].clone()
        self.icm_class_counts = counts["icm"].clone()
        self.te_class_counts = counts["te"].clone()

        self.icm_num_classes = len(ICM_CLASSES)
        self.te_num_classes = len(TE_CLASSES)

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
        def _override_weights(values, num_classes: int, head: str) -> Optional[torch.Tensor]:
            if values is None:
                return None
            if len(values) != num_classes:
                raise ValueError(f"{head} class_weights length={len(values)} does not match num_classes={num_classes}.")
            return torch.tensor(values, dtype=torch.float)

        exp_override = _override_weights(self.morph_class_weights_exp, self.exp_num_classes, "EXP")
        icm_override = _override_weights(self.morph_class_weights_icm, self.icm_num_classes, "ICM")
        te_override = _override_weights(self.morph_class_weights_te, self.te_num_classes, "TE")

        if exp_override is not None or icm_override is not None or te_override is not None or self.use_class_weights:
            self.exp_class_weight = exp_override if exp_override is not None else _compute_weights("exp", self.exp_num_classes)
            self.icm_class_weight = icm_override if icm_override is not None else _compute_weights("icm", self.icm_num_classes)
            self.te_class_weight = te_override if te_override is not None else _compute_weights("te", self.te_num_classes)
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
