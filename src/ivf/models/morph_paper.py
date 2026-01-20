"""
Paper reproduction morphology model:
- Encoder: torchvision ResNet/EfficientNet (ImageNet weights optional)
- Heads: EXP 5-class (0..4), ICM 4-class (A/B/C/ND=0..3), TE 4-class (A/B/C/ND=0..3)

Metrics for ICM/TE are later computed only on the subset where EXP>=3 for both
prediction and ground-truth, matching the reference repo scripts.

This intentionally avoids the custom MorphologyBackbone used in JointMorphNet.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import torch
from torch import nn

try:  # optional dependency
    from torchvision import models
except Exception:  # pragma: no cover
    models = None  # type: ignore

from ivf.models.heads import MorphologyHeads


PaperBackboneName = Literal["resnet50", "efficientnet_b0"]


def _require_torchvision() -> None:
    if models is None:
        raise ImportError("torchvision is required for morph_paper backbones (resnet/efficientnet).")


def _build_resnet50(pretrained: bool) -> tuple[nn.Module, int, str]:
    _require_torchvision()
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    encoder = models.resnet50(weights=weights)
    feature_dim = int(encoder.fc.in_features)
    encoder.fc = nn.Identity()
    return encoder, feature_dim, "ResNet50"


def _build_efficientnet_b0(pretrained: bool) -> tuple[nn.Module, int, str]:
    _require_torchvision()
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    encoder = models.efficientnet_b0(weights=weights)
    feature_dim = int(encoder.classifier[-1].in_features)
    encoder.classifier = nn.Identity()
    return encoder, feature_dim, "EfficientNetB0"


def build_paper_encoder(backbone: str, pretrained: bool) -> tuple[nn.Module, int, str]:
    name = str(backbone).lower()
    if name == "resnet50":
        return _build_resnet50(pretrained)
    if name == "efficientnet_b0":
        return _build_efficientnet_b0(pretrained)
    raise ValueError(f"Unsupported paper backbone: {backbone!r} (expected resnet50 or efficientnet_b0).")


class PaperMorphNet(nn.Module):
    def __init__(
        self,
        backbone: PaperBackboneName = "resnet50",
        pretrained: bool = True,
        head_hidden_dim: int = 0,
        exp_num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.encoder, feature_dim, encoder_name = build_paper_encoder(backbone, pretrained)
        self.encoder_name = encoder_name
        self.morph = MorphologyHeads(
            feature_dim,
            hidden_dim=head_hidden_dim,
            exp_classes=int(exp_num_classes),
            icm_classes=4,
            te_classes=4,
        )
        self.exp_num_classes = int(exp_num_classes)
        self.icm_num_classes = 4
        self.te_num_classes = 4

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        features = self.encoder(x)
        if isinstance(features, (tuple, list)):
            features = features[-1]
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        morph_logits = self.morph(features)
        return {"features": features, "morph": morph_logits}
