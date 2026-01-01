"""
Loss utilities for multitask IVF model.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.nn import functional as F


@dataclass
class LossWeights:
    morph: float = 1.0
    stage: float = 1.0
    quality: float = 1.0


def morphology_loss(logits: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    loss_exp = F.cross_entropy(logits["exp"], targets["exp"])
    loss_icm = F.cross_entropy(logits["icm"], targets["icm"])
    loss_te = F.cross_entropy(logits["te"], targets["te"])
    return loss_exp + loss_icm + loss_te


def stage_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


def quality_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Two-class cross-entropy for quality (good/poor).
    """
    return F.cross_entropy(logits, targets)


def compute_total_loss(outputs: Dict[str, object], targets: Dict[str, object], weights: Optional[LossWeights] = None) -> Dict[str, torch.Tensor]:
    weights = weights or LossWeights()
    losses = {}

    if "morph" in outputs and "morph" in targets:
        losses["morphology"] = morphology_loss(outputs["morph"], targets["morph"]) * weights.morph
    if "stage" in outputs and "stage" in targets:
        losses["stage"] = stage_loss(outputs["stage"], targets["stage"]) * weights.stage
    if "quality" in outputs and "quality" in targets:
        losses["quality"] = quality_loss(outputs["quality"], targets["quality"]) * weights.quality

    total = sum(losses.values()) if losses else torch.tensor(0.0, device=next(iter(outputs.values())).device)
    losses["total"] = total
    return losses
