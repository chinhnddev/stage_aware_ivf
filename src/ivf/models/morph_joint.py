"""
Joint morphology predictor with a shared encoder and three classification heads.
"""

from typing import Dict, Optional

import torch
from torch import nn

from ivf.models.encoder import ConvNeXtMini
from ivf.models.heads import MorphologyHeads


class JointMorphNet(nn.Module):
    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        feature_dim: int = 256,
        exp_num_classes: int = 5,
        icm_num_classes: int = 3,
        te_num_classes: int = 3,
        head_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else ConvNeXtMini(feature_dim=feature_dim)
        self.morph = MorphologyHeads(
            feature_dim,
            hidden_dim=head_hidden_dim,
            exp_classes=exp_num_classes,
            icm_classes=icm_num_classes,
            te_classes=te_num_classes,
        )
        self.exp_num_classes = exp_num_classes
        self.icm_num_classes = icm_num_classes
        self.te_num_classes = te_num_classes

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        features = self.encoder(x)
        morph_logits = self.morph(features)
        return {
            "features": features,
            "morph": morph_logits,
        }
