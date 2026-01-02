"""
Multi-task IVF model with shared encoder and stage-conditioned quality head.
"""

from typing import Dict

import torch
from torch import nn

from ivf.models.encoder import ConvNeXtMini
from ivf.models.heads import MorphologyHeads, StageConditionedQualityHead, StageHead


class MultiTaskEmbryoNet(nn.Module):
    def __init__(
        self,
        encoder: nn.Module = None,
        feature_dim: int = 256,
        quality_mode: str = "concat",
        quality_conditioning: str = "stage",
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else ConvNeXtMini(feature_dim=feature_dim)
        self.morph = MorphologyHeads(feature_dim)
        self.stage = StageHead(feature_dim)
        self.quality = StageConditionedQualityHead(feature_dim, mode=quality_mode)
        self.quality_conditioning = quality_conditioning

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        features = self.encoder(x)
        stage_logits = self.stage(features)
        if self.quality_conditioning == "uniform":
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
        morph_logits = self.morph(features)

        return {
            "features": features,
            "morph": morph_logits,
            "stage": stage_logits,
            "quality": quality_logits,
        }
