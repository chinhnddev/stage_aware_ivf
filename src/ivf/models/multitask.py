"""
Multi-task IVF model with shared encoder and morphology/stage-conditioned quality head.
"""

from typing import Dict

import torch
from torch import nn

from ivf.data.label_schema import EXPANSION_CLASSES, ICM_CLASSES, STAGE_CLASSES, TE_CLASSES
from ivf.models.encoder import ConvNeXtMini
from ivf.models.heads import (
    ConditionedQualityHead,
    MorphologyHeads,
    QualityHead,
    StageConditionedQualityHead,
    StageHead,
)


class MultiTaskEmbryoNet(nn.Module):
    def __init__(
        self,
        encoder: nn.Module = None,
        feature_dim: int = 256,
        quality_mode: str = "concat",
        quality_conditioning: str = "morph+stage",
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else ConvNeXtMini(feature_dim=feature_dim)
        self.morph = MorphologyHeads(feature_dim)
        self.stage = StageHead(feature_dim)
        self.q_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
        )
        self.quality_conditioning = quality_conditioning
        if quality_conditioning == "none":
            self.quality = QualityHead(feature_dim)
        elif quality_conditioning in {"stage", "uniform", "detach"}:
            self.quality = StageConditionedQualityHead(feature_dim, mode=quality_mode)
        elif quality_conditioning == "morph+stage":
            cond_dim = len(STAGE_CLASSES) + len(EXPANSION_CLASSES) + len(ICM_CLASSES) + len(TE_CLASSES)
            self.quality = ConditionedQualityHead(feature_dim, cond_dim=cond_dim, mode=quality_mode)
        else:
            raise ValueError(f"Unsupported quality_conditioning: {quality_conditioning}")

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        features = self.encoder(x)
        morph_logits = self.morph(features)
        stage_logits = self.stage(features)
        q_logits = self.q_head(features)
        q_pred = torch.sigmoid(q_logits).squeeze(-1)

        if self.quality_conditioning == "none":
            quality_logits = self.quality(features)
        elif self.quality_conditioning == "morph+stage":
            stage_probs = torch.softmax(stage_logits, dim=-1)
            exp_probs = torch.softmax(morph_logits["exp"], dim=-1)
            icm_probs = torch.softmax(morph_logits["icm"], dim=-1)
            te_probs = torch.softmax(morph_logits["te"], dim=-1)
            morph_stage_vec = torch.cat([stage_probs, exp_probs, icm_probs, te_probs], dim=-1)
            quality_logits = self.quality(features, morph_stage_vec)
        elif self.quality_conditioning == "uniform":
            num_classes = stage_logits.shape[-1]
            stage_probs = torch.full(
                (stage_logits.shape[0], num_classes),
                1.0 / num_classes,
                device=stage_logits.device,
                dtype=stage_logits.dtype,
            )
            quality_logits = self.quality(features, stage_probs=stage_probs)
        elif self.quality_conditioning == "detach":
            quality_logits = self.quality(features, stage_logits=stage_logits.detach())
        else:
            quality_logits = self.quality(features, stage_logits=stage_logits)

        return {
            "features": features,
            "morph": morph_logits,
            "stage": stage_logits,
            "quality": quality_logits,
            "q": q_pred,
        }
